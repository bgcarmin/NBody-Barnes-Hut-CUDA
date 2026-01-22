// nbody_cuda_barneshut_viz.cu - Barnes-Hut sa OpenGL vizualizacijom

#include <cuda_runtime.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GL/freeglut.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define G_CONST 5.0f
#define THETA 0.5f
#define DT 0.3f
#define SOFTENING 0.001f
#define EPSILON 1e-6f

// ============================================================================
// OCTREE NODE STRUCTURE
// ============================================================================
struct OctreeNode {
    float mass;
    float comX, comY, comZ;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    int children[8];
    int bodyIndex;
    int locked;
};

// ============================================================================
// GLOBALS FOR VISUALIZATION
// ============================================================================
int N = 10000;
int currentStep = 0;
bool paused = false;
float camDistance = 1000.0f;
float camRotX = 30.0f;
float camRotY = 45.0f;
int lastMouseX, lastMouseY;
bool mouseRotate = false;

// Device pointers
float *d_posX, *d_posY, *d_posZ;
float *d_velX, *d_velY, *d_velZ;
float *d_accX, *d_accY, *d_accZ;
float *d_mass;
OctreeNode *d_nodes;
float *d_minX, *d_minY, *d_minZ;
float *d_maxX, *d_maxY, *d_maxZ;
unsigned long long *d_morton;
int *d_indices, *d_nodeCounter;

// OpenGL buffers
GLuint vbo_positions;
cudaGraphicsResource *cuda_vbo_resource;

// ============================================================================
// MORTON CODE FOR SPATIAL SORTING
// ============================================================================
__device__ __host__ inline unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline unsigned long long morton3D(
    float x, float y, float z,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ
) {
    float nx = (x - minX) / (maxX - minX + 1e-10f);
    float ny = (y - minY) / (maxY - minY + 1e-10f);
    float nz = (z - minZ) / (maxZ - minZ + 1e-10f);
    nx = fminf(fmaxf(nx * 1024.0f, 0.0f), 1023.0f);
    ny = fminf(fmaxf(ny * 1024.0f, 0.0f), 1023.0f);
    nz = fminf(fmaxf(nz * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)nx);
    unsigned int yy = expandBits((unsigned int)ny);
    unsigned int zz = expandBits((unsigned int)nz);
    return ((unsigned long long)xx << 2) | ((unsigned long long)yy << 1) | (unsigned long long)zz;
}

// ============================================================================
// KERNEL 1: BOUNDING BOX
// ============================================================================
__global__ void computeBoundingBoxKernel(
    const float *posX, const float *posY, const float *posZ,
    int numBodies,
    float *minX, float *minY, float *minZ,
    float *maxX, float *maxY, float *maxZ
) {
    __shared__ float sMinX[THREADS_PER_BLOCK];
    __shared__ float sMinY[THREADS_PER_BLOCK];
    __shared__ float sMinZ[THREADS_PER_BLOCK];
    __shared__ float sMaxX[THREADS_PER_BLOCK];
    __shared__ float sMaxY[THREADS_PER_BLOCK];
    __shared__ float sMaxZ[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float tMinX = 1e30f, tMinY = 1e30f, tMinZ = 1e30f;
    float tMaxX = -1e30f, tMaxY = -1e30f, tMaxZ = -1e30f;

    for (int i = idx; i < numBodies; i += stride) {
        tMinX = fminf(tMinX, posX[i]);
        tMinY = fminf(tMinY, posY[i]);
        tMinZ = fminf(tMinZ, posZ[i]);
        tMaxX = fmaxf(tMaxX, posX[i]);
        tMaxY = fmaxf(tMaxY, posY[i]);
        tMaxZ = fmaxf(tMaxZ, posZ[i]);
    }

    sMinX[tid] = tMinX; sMinY[tid] = tMinY; sMinZ[tid] = tMinZ;
    sMaxX[tid] = tMaxX; sMaxY[tid] = tMaxY; sMaxZ[tid] = tMaxZ;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sMinX[tid] = fminf(sMinX[tid], sMinX[tid + s]);
            sMinY[tid] = fminf(sMinY[tid], sMinY[tid + s]);
            sMinZ[tid] = fminf(sMinZ[tid], sMinZ[tid + s]);
            sMaxX[tid] = fmaxf(sMaxX[tid], sMaxX[tid + s]);
            sMaxY[tid] = fmaxf(sMaxY[tid], sMaxY[tid + s]);
            sMaxZ[tid] = fmaxf(sMaxZ[tid], sMaxZ[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin((int*)minX, __float_as_int(sMinX[0]));
        atomicMin((int*)minY, __float_as_int(sMinY[0]));
        atomicMin((int*)minZ, __float_as_int(sMinZ[0]));
        atomicMax((int*)maxX, __float_as_int(sMaxX[0]));
        atomicMax((int*)maxY, __float_as_int(sMaxY[0]));
        atomicMax((int*)maxZ, __float_as_int(sMaxZ[0]));
    }
}

// ============================================================================
// KERNEL 2: SIMPLE OCTREE
// ============================================================================
__device__ inline int getOctant(float px, float py, float pz,
                                float midX, float midY, float midZ) {
    int octant = 0;
    if (px >= midX) octant |= 1;
    if (py >= midY) octant |= 2;
    if (pz >= midZ) octant |= 4;
    return octant;
}

__global__ void buildOctreeSimpleKernel(
    const float *posX, const float *posY, const float *posZ,
    const unsigned long long *mortonCodes,
    const int *sortedIndices,
    OctreeNode *nodes,
    int numBodies,
    int *nodeCounter,
    float rootMinX, float rootMinY, float rootMinZ,
    float rootMaxX, float rootMaxY, float rootMaxZ
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *nodeCounter = 0;
        nodes[0].minX = rootMinX;
        nodes[0].minY = rootMinY;
        nodes[0].minZ = rootMinZ;
        nodes[0].maxX = rootMaxX;
        nodes[0].maxY = rootMaxY;
        nodes[0].maxZ = rootMaxZ;
        nodes[0].mass = 0.0f;
        nodes[0].bodyIndex = -1;
        nodes[0].locked = 0;
        for (int i = 0; i < 8; i++) nodes[0].children[i] = -1;

        for (int i = 0; i < numBodies; i++) {
            int bodyIdx = sortedIndices[i];
            float px = posX[bodyIdx];
            float py = posY[bodyIdx];
            float pz = posZ[bodyIdx];
            int nodeIdx = 0;
            int depth = 0;

            while (depth < 20) {
                OctreeNode *node = &nodes[nodeIdx];
                float midX = (node->minX + node->maxX) * 0.5f;
                float midY = (node->minY + node->maxY) * 0.5f;
                float midZ = (node->minZ + node->maxZ) * 0.5f;
                int octant = getOctant(px, py, pz, midX, midY, midZ);
                int childIdx = node->children[octant];

                if (childIdx == -1) {
                    node->children[octant] = bodyIdx;
                    break;
                } else if (childIdx < numBodies) {
                    int newIdx = numBodies + atomicAdd(nodeCounter, 1);
                    OctreeNode *newNode = &nodes[newIdx];
                    newNode->minX = (octant & 1) ? midX : node->minX;
                    newNode->maxX = (octant & 1) ? node->maxX : midX;
                    newNode->minY = (octant & 2) ? midY : node->minY;
                    newNode->maxY = (octant & 2) ? node->maxY : midY;
                    newNode->minZ = (octant & 4) ? midZ : node->minZ;
                    newNode->maxZ = (octant & 4) ? node->maxZ : midZ;
                    newNode->mass = 0.0f;
                    newNode->bodyIndex = -1;
                    for (int k = 0; k < 8; k++) newNode->children[k] = -1;

                    float oldPx = posX[childIdx];
                    float oldPy = posY[childIdx];
                    float oldPz = posZ[childIdx];
                    float newMidX = (newNode->minX + newNode->maxX) * 0.5f;
                    float newMidY = (newNode->minY + newNode->maxY) * 0.5f;
                    float newMidZ = (newNode->minZ + newNode->maxZ) * 0.5f;
                    int oldOct = getOctant(oldPx, oldPy, oldPz, newMidX, newMidY, newMidZ);
                    newNode->children[oldOct] = childIdx;
                    node->children[octant] = newIdx;
                    nodeIdx = newIdx;
                } else {
                    nodeIdx = childIdx;
                }
                depth++;
            }
        }
    }
}

// ============================================================================
// KERNEL 3: CENTER OF MASS
// ============================================================================
__global__ void computeCenterOfMassKernel(
    OctreeNode *nodes,
    const float *bodyPosX, const float *bodyPosY, const float *bodyPosZ,
    const float *bodyMass,
    int numBodies,
    int totalNodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = totalNodes - 1 - idx; i >= 0; i -= stride) {
        if (i >= numBodies) {
            OctreeNode *node = &nodes[i];
            float totalMass = 0.0f;
            float comX = 0.0f, comY = 0.0f, comZ = 0.0f;

            for (int k = 0; k < 8; k++) {
                int childIdx = node->children[k];
                if (childIdx >= 0) {
                    if (childIdx < numBodies) {
                        float m = bodyMass[childIdx];
                        totalMass += m;
                        comX += bodyPosX[childIdx] * m;
                        comY += bodyPosY[childIdx] * m;
                        comZ += bodyPosZ[childIdx] * m;
                    } else {
                        OctreeNode *child = &nodes[childIdx];
                        float m = child->mass;
                        totalMass += m;
                        comX += child->comX * m;
                        comY += child->comY * m;
                        comZ += child->comZ * m;
                    }
                }
            }

            if (totalMass > 0.0f) {
                float invMass = 1.0f / totalMass;
                node->comX = comX * invMass;
                node->comY = comY * invMass;
                node->comZ = comZ * invMass;
            }
            node->mass = totalMass;
        }
    }
}

// ============================================================================
// KERNEL 4: BARNES-HUT FORCE CALCULATION
// ============================================================================
__global__ void computeForceBarnesHutKernel(
    const float *posX, const float *posY, const float *posZ,
    const float *mass,
    float *accX, float *accY, float *accZ,
    const OctreeNode *nodes,
    int numBodies,
    float theta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const OctreeNode *node = &nodes[nodeIdx];

        if (node->mass <= 0.0f) continue;

        float dx = node->comX - px;
        float dy = node->comY - py;
        float dz = node->comZ - pz;
        float distSq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
        float dist = sqrtf(distSq);
        float width = node->maxX - node->minX;

        bool farEnough = (width / (dist + EPSILON)) < theta;
        bool isSelf = (nodeIdx < numBodies && nodeIdx == i);

        if ((farEnough || nodeIdx < numBodies) && !isSelf) {
            float invDist = rsqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;
            float f = G_CONST * node->mass * invDist3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        } else if (!isSelf) {
            for (int k = 0; k < 8; k++) {
                if (node->children[k] >= 0 && stackPtr < 63) {
                    stack[stackPtr++] = node->children[k];
                }
            }
        }
    }

    accX[i] = ax;
    accY[i] = ay;
    accZ[i] = az;
}

// ============================================================================
// KERNEL 5: INTEGRATION
// ============================================================================
__global__ void integrateKernel(
    float *posX, float *posY, float *posZ,
    float *velX, float *velY, float *velZ,
    const float *accX, const float *accY, const float *accZ,
    int numBodies, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    velX[i] += accX[i] * dt;
    velY[i] += accY[i] * dt;
    velZ[i] += accZ[i] * dt;

    posX[i] += velX[i] * dt;
    posY[i] += velY[i] * dt;
    posZ[i] += velZ[i] * dt;
}

// ============================================================================
// KERNEL 6: UPDATE VBO (novi kernel za vizualizaciju)
// ============================================================================
__global__ void updateVBOKernel(
    float *vbo,
    const float *posX, const float *posY, const float *posZ,
    int numBodies
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    vbo[i * 3 + 0] = posX[i];
    vbo[i * 3 + 1] = posY[i];
    vbo[i * 3 + 2] = posZ[i];
}

// ============================================================================
// MORTON CODE SORTING KERNEL
// ============================================================================
__global__ void computeMortonCodesKernel(
    const float *posX, const float *posY, const float *posZ,
    unsigned long long *mortonCodes, int *indices,
    int numBodies,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    mortonCodes[i] = morton3D(posX[i], posY[i], posZ[i],
                              minX, minY, minZ, maxX, maxY, maxZ);
    indices[i] = i;
}

// ============================================================================
// THRUST SORTING
// ============================================================================
void sortByMortonCode(unsigned long long *mortonCodes, int *indices, int N) {
    thrust::device_ptr<unsigned long long> keys(mortonCodes);
    thrust::device_ptr<int> values(indices);
    thrust::sort_by_key(keys, keys + N, values);
}

// ============================================================================
// ERROR CHECKING
// ============================================================================
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// SIMULATION STEP
// ============================================================================
void simulationStep() {
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Reset bounds
    float inf = 1e30f, ninf = -1e30f;
    cudaMemcpy(d_minX, &inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minY, &inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minZ, &inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxX, &ninf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxY, &ninf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxZ, &ninf, sizeof(float), cudaMemcpyHostToDevice);

    // Compute bounding box
    computeBoundingBoxKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_posX, d_posY, d_posZ, N,
        d_minX, d_minY, d_minZ, d_maxX, d_maxY, d_maxZ
    );
    cudaDeviceSynchronize();

    float h_minX, h_minY, h_minZ, h_maxX, h_maxY, h_maxZ;
    cudaMemcpy(&h_minX, d_minX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minY, d_minY, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minZ, d_minZ, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_maxX, d_maxX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_maxY, d_maxY, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_maxZ, d_maxZ, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute Morton codes
    computeMortonCodesKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_posX, d_posY, d_posZ, d_morton, d_indices, N,
        h_minX, h_minY, h_minZ, h_maxX, h_maxY, h_maxZ
    );
    cudaDeviceSynchronize();

    // Sort by Morton code
    sortByMortonCode(d_morton, d_indices, N);

    // Build octree
    cudaMemset(d_nodeCounter, 0, sizeof(int));
    buildOctreeSimpleKernel<<<1, 1>>>(
        d_posX, d_posY, d_posZ, d_morton, d_indices,
        d_nodes, N, d_nodeCounter,
        h_minX, h_minY, h_minZ, h_maxX, h_maxY, h_maxZ
    );
    cudaDeviceSynchronize();

    int h_nodeCounter;
    cudaMemcpy(&h_nodeCounter, d_nodeCounter, sizeof(int), cudaMemcpyDeviceToHost);
    int totalNodes = N + h_nodeCounter;

    // Compute center of mass
    computeCenterOfMassKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_nodes, d_posX, d_posY, d_posZ, d_mass, N, totalNodes
    );
    cudaDeviceSynchronize();

    // Compute forces
    computeForceBarnesHutKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_posX, d_posY, d_posZ, d_mass,
        d_accX, d_accY, d_accZ, d_nodes, N, THETA
    );
    cudaDeviceSynchronize();

    // Integrate
    integrateKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
        d_posX, d_posY, d_posZ,
        d_velX, d_velY, d_velZ,
        d_accX, d_accY, d_accZ, N, DT
    );
    cudaDeviceSynchronize();

    currentStep++;
}

// ============================================================================
// OPENGL DISPLAY
// ============================================================================
void display() {
    if (!paused) {
        simulationStep();
    }

    // Map VBO
    float *d_vbo_ptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);

    // Update VBO with positions
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    updateVBOKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_vbo_ptr, d_posX, d_posY, d_posZ, N);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    // Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef(0.0f, 0.0f, -camDistance);
    glRotatef(camRotX, 1.0f, 0.0f, 0.0f);
    glRotatef(camRotY, 0.0f, 1.0f, 0.0f);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glColor3f(1.0f, 1.0f, 1.0f);
    glPointSize(2.0f);
    glDrawArrays(GL_POINTS, 0, N);

    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

// ============================================================================
// OPENGL CALLBACKS
// ============================================================================
void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)w / (double)h, 1.0, 10000.0);
    glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // ESC
            exit(0);
            break;
        case ' ':
            paused = !paused;
            printf("Simulation %s\n", paused ? "PAUSED" : "RESUMED");
            break;
        case 'r':
            camDistance = 1000.0f;
            camRotX = 30.0f;
            camRotY = 45.0f;
            printf("Camera reset\n");
            break;
    }
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouseRotate = true;
            lastMouseX = x;
            lastMouseY = y;
        } else {
            mouseRotate = false;
        }
    }
}

void motion(int x, int y) {
    if (mouseRotate) {
        camRotY += (x - lastMouseX) * 0.5f;
        camRotX += (y - lastMouseY) * 0.5f;
        lastMouseX = x;
        lastMouseY = y;
    }
}

void mouseWheel(int button, int dir, int x, int y) {
    if (dir > 0) {
        camDistance *= 0.9f;
    } else {
        camDistance *= 1.1f;
    }
    camDistance = fmaxf(100.0f, fminf(camDistance, 5000.0f));
}

// ============================================================================
// INIT OPENGL
// ============================================================================
void initOpenGL(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1280, 720);
    glutCreateWindow("Barnes-Hut N-Body CUDA+OpenGL");

    glewInit();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMouseWheelFunc(mouseWheel);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Create VBO
    glGenBuffers(1, &vbo_positions);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
    glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo_positions, cudaGraphicsMapFlagsWriteDiscard);
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char **argv) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device found!\n");
        return 1;
    }

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("========================================\n");
    printf("Barnes-Hut N-Body with OpenGL\n");
    printf("========================================\n");
    printf("Device: %s\n", prop.name);
    printf("Bodies: %d\n", N);
    printf("Theta: %.2f\n", THETA);
    printf("========================================\n");
    printf("Controls:\n");
    printf("  SPACE - Pause/Resume\n");
    printf("  Mouse - Rotate view\n");
    printf("  Wheel - Zoom in/out\n");
    printf("  R - Reset camera\n");
    printf("  ESC - Exit\n");
    printf("========================================\n\n");

    // Allocate host
    float *h_posX = new float[N];
    float *h_posY = new float[N];
    float *h_posZ = new float[N];
    float *h_velX = new float[N];
    float *h_velY = new float[N];
    float *h_velZ = new float[N];
    float *h_mass = new float[N];

    // Initialize
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    printf("Initializing...\n");
    for (int i = 0; i < N; i++) {
        float angle = dist(rng) * 2.0f * (float)M_PI;
        float r = dist(rng) * 400.0f;
        h_posX[i] = r * cosf(angle);
        h_posY[i] = r * sinf(angle);
        h_posZ[i] = (dist(rng) - 0.5f) * 100.0f;
        h_mass[i] = 1000.0f;
        h_velX[i] = -h_posY[i] * 0.01f;
        h_velY[i] = h_posX[i] * 0.01f;
        h_velZ[i] = 0.0f;
    }

    // Allocate device
    checkCudaError(cudaMalloc(&d_posX, N * sizeof(float)), "malloc posX");
    checkCudaError(cudaMalloc(&d_posY, N * sizeof(float)), "malloc posY");
    checkCudaError(cudaMalloc(&d_posZ, N * sizeof(float)), "malloc posZ");
    checkCudaError(cudaMalloc(&d_velX, N * sizeof(float)), "malloc velX");
    checkCudaError(cudaMalloc(&d_velY, N * sizeof(float)), "malloc velY");
    checkCudaError(cudaMalloc(&d_velZ, N * sizeof(float)), "malloc velZ");
    checkCudaError(cudaMalloc(&d_accX, N * sizeof(float)), "malloc accX");
    checkCudaError(cudaMalloc(&d_accY, N * sizeof(float)), "malloc accY");
    checkCudaError(cudaMalloc(&d_accZ, N * sizeof(float)), "malloc accZ");
    checkCudaError(cudaMalloc(&d_mass, N * sizeof(float)), "malloc mass");

    checkCudaError(cudaMemcpy(d_posX, h_posX, N * sizeof(float), cudaMemcpyHostToDevice), "copy posX");
    checkCudaError(cudaMemcpy(d_posY, h_posY, N * sizeof(float), cudaMemcpyHostToDevice), "copy posY");
    checkCudaError(cudaMemcpy(d_posZ, h_posZ, N * sizeof(float), cudaMemcpyHostToDevice), "copy posZ");
    checkCudaError(cudaMemcpy(d_velX, h_velX, N * sizeof(float), cudaMemcpyHostToDevice), "copy velX");
    checkCudaError(cudaMemcpy(d_velY, h_velY, N * sizeof(float), cudaMemcpyHostToDevice), "copy velY");
    checkCudaError(cudaMemcpy(d_velZ, h_velZ, N * sizeof(float), cudaMemcpyHostToDevice), "copy velZ");
    checkCudaError(cudaMemcpy(d_mass, h_mass, N * sizeof(float), cudaMemcpyHostToDevice), "copy mass");

    // Octree
    const int MAX_NODES = N * 4;
    checkCudaError(cudaMalloc(&d_nodes, MAX_NODES * sizeof(OctreeNode)), "malloc nodes");

    checkCudaError(cudaMalloc(&d_minX, sizeof(float)), "malloc minX");
    checkCudaError(cudaMalloc(&d_minY, sizeof(float)), "malloc minY");
    checkCudaError(cudaMalloc(&d_minZ, sizeof(float)), "malloc minZ");
    checkCudaError(cudaMalloc(&d_maxX, sizeof(float)), "malloc maxX");
    checkCudaError(cudaMalloc(&d_maxY, sizeof(float)), "malloc maxY");
    checkCudaError(cudaMalloc(&d_maxZ, sizeof(float)), "malloc maxZ");

    checkCudaError(cudaMalloc(&d_morton, N * sizeof(unsigned long long)), "malloc morton");
    checkCudaError(cudaMalloc(&d_indices, N * sizeof(int)), "malloc indices");
    checkCudaError(cudaMalloc(&d_nodeCounter, sizeof(int)), "malloc counter");

    delete[] h_posX; delete[] h_posY; delete[] h_posZ;
    delete[] h_velX; delete[] h_velY; delete[] h_velZ;
    delete[] h_mass;

    // Init OpenGL
    initOpenGL(argc, argv);

    printf("Starting visualization...\n\n");
    glutMainLoop();

    // Cleanup
    cudaFree(d_posX); cudaFree(d_posY); cudaFree(d_posZ);
    cudaFree(d_velX); cudaFree(d_velY); cudaFree(d_velZ);
    cudaFree(d_accX); cudaFree(d_accY); cudaFree(d_accZ);
    cudaFree(d_mass); cudaFree(d_nodes);
    cudaFree(d_minX); cudaFree(d_minY); cudaFree(d_minZ);
    cudaFree(d_maxX); cudaFree(d_maxY); cudaFree(d_maxZ);
    cudaFree(d_morton); cudaFree(d_indices); cudaFree(d_nodeCounter);

    return 0;
}
