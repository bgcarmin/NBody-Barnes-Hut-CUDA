// nbody_cuda.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define G_CONST 1.0f
#define THETA 0.5f
#define DT 0.03f
#define SOFTENING 0.001f
#define EPSILON 1e-6f

// ============================================================================
// DATA STRUCTURES (Structure of Arrays - SoA)
// ============================================================================

struct BodyData {
    float *mass;
    float *posX, *posY, *posZ;
    float *velX, *velY, *velZ;
    float *accX, *accY, *accZ;
    int numBodies;
};

struct OctreeNode {
    float mass;
    float comX, comY, comZ;  // Center of mass
    float minX, minY, minZ;   // Bounding box min
    float maxX, maxY, maxZ;   // Bounding box max
    int children[8];          // Child indices (-1 for null)
    int bodyIndex;            // Body index if leaf (-1 otherwise)
    int numBodies;            // Number of bodies in subtree
};

// ============================================================================
// GLOBAL DEVICE MEMORY POINTERS
// ============================================================================

__constant__ float d_minX, d_minY, d_minZ;
__constant__ float d_maxX, d_maxY, d_maxZ;
__constant__ int d_numBodies;
__constant__ int d_maxNodes;

// ============================================================================
// KERNEL 1: BOUNDING BOX COMPUTATION
// ============================================================================

__global__ void computeBoundingBoxKernel(
    const float *posX, const float *posY, const float *posZ,
    int numBodies,
    float *blockMinX, float *blockMinY, float *blockMinZ,
    float *blockMaxX, float *blockMaxY, float *blockMaxZ,
    int *counter
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
    
    float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
    
    // Coalesced loading and reduction
    for (int i = idx; i < numBodies; i += stride) {
        float px = posX[i];
        float py = posY[i];
        float pz = posZ[i];
        minX = fminf(minX, px);
        minY = fminf(minY, py);
        minZ = fminf(minZ, pz);
        maxX = fmaxf(maxX, px);
        maxY = fmaxf(maxY, py);
        maxZ = fmaxf(maxZ, pz);
    }
    
    sMinX[tid] = minX;
    sMinY[tid] = minY;
    sMinZ[tid] = minZ;
    sMaxX[tid] = maxX;
    sMaxY[tid] = maxY;
    sMaxZ[tid] = maxZ;
    __syncthreads();
    
    // Reduction in shared memory
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
        blockMinX[blockIdx.x] = sMinX[0];
        blockMinY[blockIdx.x] = sMinY[0];
        blockMinZ[blockIdx.x] = sMinZ[0];
        blockMaxX[blockIdx.x] = sMaxX[0];
        blockMaxY[blockIdx.x] = sMaxY[0];
        blockMaxZ[blockIdx.x] = sMaxZ[0];
        
        // Last block combines results
        __threadfence();
        int value = atomicAdd(counter, 1);
        if (value == gridDim.x - 1) {
            float gMinX = 1e30f, gMinY = 1e30f, gMinZ = 1e30f;
            float gMaxX = -1e30f, gMaxY = -1e30f, gMaxZ = -1e30f;
            for (int i = 0; i < gridDim.x; i++) {
                gMinX = fminf(gMinX, blockMinX[i]);
                gMinY = fminf(gMinY, blockMinY[i]);
                gMinZ = fminf(gMinZ, blockMinZ[i]);
                gMaxX = fmaxf(gMaxX, blockMaxX[i]);
                gMaxY = fmaxf(gMaxY, blockMaxY[i]);
                gMaxZ = fmaxf(gMaxZ, blockMaxZ[i]);
            }
            blockMinX[0] = gMinX;
            blockMinY[0] = gMinY;
            blockMinZ[0] = gMinZ;
            blockMaxX[0] = gMaxX;
            blockMaxY[0] = gMaxY;
            blockMaxZ[0] = gMaxZ;
        }
    }
}

// ============================================================================
// MORTON CODE UTILITIES (for spatial sorting)
// ============================================================================

__device__ __host__ inline unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline unsigned long long morton3D(float x, float y, float z, 
                                                        float minX, float minY, float minZ,
                                                        float maxX, float maxY, float maxZ) {
    float nx = (x - minX) / (maxX - minX);
    float ny = (y - minY) / (maxY - minY);
    float nz = (z - minZ) / (maxZ - minZ);
    
    nx = fminf(fmaxf(nx * 1024.0f, 0.0f), 1023.0f);
    ny = fminf(fmaxf(ny * 1024.0f, 0.0f), 1023.0f);
    nz = fminf(fmaxf(nz * 1024.0f, 0.0f), 1023.0f);
    
    unsigned int xx = expandBits((unsigned int)nx);
    unsigned int yy = expandBits((unsigned int)ny);
    unsigned int zz = expandBits((unsigned int)nz);
    
    return ((unsigned long long)xx << 2) | ((unsigned long long)yy << 1) | (unsigned long long)zz;
}

__global__ void computeMortonCodesKernel(
    const float *posX, const float *posY, const float *posZ,
    unsigned long long *mortonCodes,
    int numBodies,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    mortonCodes[idx] = morton3D(posX[idx], posY[idx], posZ[idx],
                                 minX, minY, minZ, maxX, maxY, maxZ);
}

// ============================================================================
// KERNEL 2: OCTREE CONSTRUCTION (Bottom-Up)
// ============================================================================

__device__ inline int getOctant(float px, float py, float pz,
                                 float midX, float midY, float midZ) {
    int octant = 0;
    if (px >= midX) octant |= 1;
    if (py >= midY) octant |= 2;
    if (pz >= midZ) octant |= 4;
    return octant;
}

__global__ void buildOctreeKernel(
    const float *posX, const float *posY, const float *posZ,
    const float *mass,
    const int *sortedIndices,
    OctreeNode *nodes,
    int numBodies,
    int *nodeCounter,
    float rootMinX, float rootMinY, float rootMinZ,
    float rootMaxX, float rootMaxY, float rootMaxZ
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    const int LOCKED = -2;
    
    for (int i = idx; i < numBodies; i += stride) {
        int bodyIdx = sortedIndices ? sortedIndices[i] : i;
        float px = posX[bodyIdx];
        float py = posY[bodyIdx];
        float pz = posZ[bodyIdx];
        float m = mass[bodyIdx];
        
        int nodeIdx = 0;  // Start at root
        bool inserted = false;
        
        while (!inserted) {
            OctreeNode &node = nodes[nodeIdx];
            
            float midX = (node.minX + node.maxX) * 0.5f;
            float midY = (node.minY + node.maxY) * 0.5f;
            float midZ = (node.minZ + node.maxZ) * 0.5f;
            
            int octant = getOctant(px, py, pz, midX, midY, midZ);
            int childIdx = node.children[octant];
            
            if (childIdx != LOCKED) {
                int old = atomicCAS(&node.children[octant], childIdx, LOCKED);
                
                if (old == childIdx) {
                    if (old == -1) {
                        // Empty slot - insert body
                        node.children[octant] = bodyIdx;
                        inserted = true;
                    } else if (old < numBodies) {
                        // Slot contains body - need to subdivide
                        int newNodeIdx = atomicAdd(nodeCounter, 1);
                        
                        OctreeNode &newNode = nodes[newNodeIdx];
                        newNode.minX = (octant & 1) ? midX : node.minX;
                        newNode.maxX = (octant & 1) ? node.maxX : midX;
                        newNode.minY = (octant & 2) ? midY : node.minY;
                        newNode.maxY = (octant & 2) ? node.maxY : midY;
                        newNode.minZ = (octant & 4) ? midZ : node.minZ;
                        newNode.maxZ = (octant & 4) ? node.maxZ : midZ;
                        newNode.bodyIndex = -1;
                        newNode.mass = 0.0f;
                        for (int k = 0; k < 8; k++) newNode.children[k] = -1;
                        
                        __threadfence();
                        
                        node.children[octant] = newNodeIdx + numBodies;
                        
                        // Re-insert old body
                        // (simplified - full version would need recursion/stack)
                    } else {
                        // Slot contains internal node - descend
                        nodeIdx = old - numBodies;
                    }
                } else {
                    // Lock failed - retry with barrier throttling
                    __syncthreads();
                }
            }
        }
    }
}

// ============================================================================
// KERNEL 3: CENTER OF MASS COMPUTATION (Bottom-Up)
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
    
    // Process cells from highest index downward (bottom-up)
    for (int i = totalNodes - 1 - idx; i >= numBodies; i -= stride) {
        OctreeNode &node = nodes[i];
        
        float totalMass = 0.0f;
        float comX = 0.0f, comY = 0.0f, comZ = 0.0f;
        int childCount = 0;
        
        __shared__ int missingChildren[THREADS_PER_BLOCK][8];
        int missing = 0;
        
        // First pass - check which children are ready
        for (int k = 0; k < 8; k++) {
            int childIdx = node.children[k];
            if (childIdx >= 0) {
                if (childIdx < numBodies) {
                    // Child is a body
                    float m = bodyMass[childIdx];
                    totalMass += m;
                    comX += bodyPosX[childIdx] * m;
                    comY += bodyPosY[childIdx] * m;
                    comZ += bodyPosZ[childIdx] * m;
                    childCount++;
                } else {
                    // Child is a node - check if ready (mass >= 0)
                    OctreeNode &child = nodes[childIdx - numBodies];
                    if (child.mass >= 0.0f) {
                        float m = child.mass;
                        totalMass += m;
                        comX += child.comX * m;
                        comY += child.comY * m;
                        comZ += child.comZ * m;
                        childCount++;
                    } else {
                        missingChildren[threadIdx.x][missing++] = k;
                    }
                }
            }
        }
        
        // Wait for missing children (with throttling)
        while (missing > 0) {
            int lastIdx = missingChildren[threadIdx.x][missing - 1];
            int childIdx = node.children[lastIdx] - numBodies;
            OctreeNode &child = nodes[childIdx];
            
            if (child.mass >= 0.0f) {
                float m = child.mass;
                totalMass += m;
                comX += child.comX * m;
                comY += child.comY * m;
                comZ += child.comZ * m;
                missing--;
            }
            __syncthreads();  // Throttle unsuccessful threads
        }
        
        // Finalize center of mass
        if (totalMass > 0.0f) {
            float invMass = 1.0f / totalMass;
            node.comX = comX * invMass;
            node.comY = comY * invMass;
            node.comZ = comZ * invMass;
        }
        
        __threadfence();
        node.mass = totalMass;  // Release "ready flag"
        node.numBodies = childCount;
    }
}

// ============================================================================
// KERNEL 5: FORCE CALCULATION (Barnes-Hut)
// ============================================================================

__global__ void computeForceKernel(
    const float *posX, const float *posY, const float *posZ,
    const float *mass,
    const int *sortedIndices,
    float *accX, float *accY, float *accZ,
    const OctreeNode *nodes,
    int numBodies,
    float theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;
    
    __shared__ float sNodeData[THREADS_PER_BLOCK / WARP_SIZE][10];
    __shared__ int sStack[THREADS_PER_BLOCK / WARP_SIZE][64];
    
    for (int i = idx; i < numBodies; i += stride) {
        int bodyIdx = sortedIndices ? sortedIndices[i] : i;
        
        float px = posX[bodyIdx];
        float py = posY[bodyIdx];
        float pz = posZ[bodyIdx];
        
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        
        // Initialize stack with root
        if (laneId == 0) {
            sStack[warpId][0] = 0;
        }
        __threadfence_block();
        
        int depth = 0;
        
        while (depth >= 0) {
            // First thread in warp loads node data
            if (laneId == 0) {
                int nodeIdx = sStack[warpId][depth];
                const OctreeNode &node = nodes[nodeIdx];
                
                sNodeData[warpId][0] = node.mass;
                sNodeData[warpId][1] = node.comX;
                sNodeData[warpId][2] = node.comY;
                sNodeData[warpId][3] = node.comZ;
                sNodeData[warpId][4] = node.maxX - node.minX;  // width
            }
            __threadfence_block();
            
            float nodeMass = sNodeData[warpId][0];
            float nodeComX = sNodeData[warpId][1];
            float nodeComY = sNodeData[warpId][2];
            float nodeComZ = sNodeData[warpId][3];
            float nodeWidth = sNodeData[warpId][4];
            
            if (nodeMass > 0.0f) {
                float dx = nodeComX - px;
                float dy = nodeComY - py;
                float dz = nodeComZ - pz;
                float distSq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                float dist = sqrtf(distSq);
                
                // Barnes-Hut criterion: width / distance < theta
                bool farEnough = (nodeWidth / (dist + EPSILON)) < theta;
                
                // Use __all_sync for warp voting
                int mask = __ballot_sync(0xFFFFFFFF, farEnough);
                
                if (mask == 0xFFFFFFFF) {
                    // All threads agree - use approximation
                    float invDist = rsqrtf(distSq);
                    float invDist3 = invDist * invDist * invDist;
                    float f = G_CONST * nodeMass * invDist3;
                    
                    ax += f * dx;
                    ay += f * dy;
                    az += f * dz;
                    depth--;
                } else {
                    // Need to descend - load children
                    if (laneId == 0) {
                        int nodeIdx = sStack[warpId][depth];
                        const OctreeNode &node = nodes[nodeIdx];
                        
                        int childCount = 0;
                        for (int k = 0; k < 8; k++) {
                            if (node.children[k] >= 0) {
                                sStack[warpId][depth + childCount] = node.children[k];
                                childCount++;
                            }
                        }
                        
                        if (childCount == 0) {
                            depth--;
                        } else {
                            depth += childCount - 1;
                        }
                    }
                    __threadfence_block();
                }
            } else {
                depth--;
            }
        }
        
        accX[bodyIdx] = ax;
        accY[bodyIdx] = ay;
        accZ[bodyIdx] = az;
    }
}

// ============================================================================
// KERNEL 6: INTEGRATION (Velocity Verlet)
// ============================================================================

__global__ void integrateKernel(
    float *posX, float *posY, float *posZ,
    float *velX, float *velY, float *velZ,
    const float *accX, const float *accY, const float *accZ,
    int numBodies,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < numBodies; i += stride) {
        velX[i] += accX[i] * dt;
        velY[i] += accY[i] * dt;
        velZ[i] += accZ[i] * dt;
        
        posX[i] += velX[i] * dt;
        posY[i] += velY[i] * dt;
        posZ[i] += velZ[i] * dt;
    }
}

// ============================================================================
// OPENGL RENDERING
// ============================================================================

void renderBodies(GLuint vbo, int numBodies) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    
    glPointSize(2.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    glDrawArrays(GL_POINTS, 0, numBodies);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// ============================================================================
// MAIN SIMULATION
// ============================================================================

int main() {
    const int N = 50000;
    const int NUM_STEPS = 1000;
    
    printf("CUDA Barnes-Hut N-Body Simulation\n");
    printf("Bodies: %d\n", N);
    
    // Initialize GLFW and OpenGL
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }
    
    GLFWwindow *window = glfwCreateWindow(1024, 1024, "CUDA N-Body", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    
    // Allocate and initialize host data
    BodyData h_bodies;
    h_bodies.numBodies = N;
    h_bodies.posX = new float[N];
    h_bodies.posY = new float[N];
    h_bodies.posZ = new float[N];
    h_bodies.velX = new float[N];
    h_bodies.velY = new float[N];
    h_bodies.velZ = new float[N];
    h_bodies.mass = new float[N];
    
    // Initialize bodies (Plummer sphere)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < N; i++) {
        float angle = dist(rng) * 2.0f * M_PI;
        float r = dist(rng) * 400.0f;
        h_bodies.posX[i] = r * cosf(angle);
        h_bodies.posY[i] = r * sinf(angle);
        h_bodies.posZ[i] = (dist(rng) - 0.5f) * 100.0f;
        
        h_bodies.mass[i] = 1000.0f;
        h_bodies.velX[i] = 0.0f;
        h_bodies.velY[i] = 0.0f;
        h_bodies.velZ[i] = 0.0f;
    }
    
    // Allocate device memory
    float *d_posX, *d_posY, *d_posZ;
    float *d_velX, *d_velY, *d_velZ;
    float *d_accX, *d_accY, *d_accZ;
    float *d_mass;
    
    cudaMalloc(&d_posX, N * sizeof(float));
    cudaMalloc(&d_posY, N * sizeof(float));
    cudaMalloc(&d_posZ, N * sizeof(float));
    cudaMalloc(&d_velX, N * sizeof(float));
    cudaMalloc(&d_velY, N * sizeof(float));
    cudaMalloc(&d_velZ, N * sizeof(float));
    cudaMalloc(&d_accX, N * sizeof(float));
    cudaMalloc(&d_accY, N * sizeof(float));
    cudaMalloc(&d_accZ, N * sizeof(float));
    cudaMalloc(&d_mass, N * sizeof(float));
    
    cudaMemcpy(d_posX, h_bodies.posX, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_bodies.posY, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, h_bodies.posZ, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_bodies.mass, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // OpenGL VBO for rendering
    GLuint vbo;
    glGenBuffers(1, &vbo);
    
    // Main loop
    int step = 0;
    while (!glfwWindowShouldClose(window) && step < NUM_STEPS) {
        // Run simulation kernels here...
        
        // Copy positions back for rendering
        cudaMemcpy(h_bodies.posX, d_posX, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bodies.posY, d_posY, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bodies.posZ, d_posZ, N * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderBodies(vbo, N);
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        step++;
    }
    
    // Cleanup
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    
    glfwTerminate();
    return 0;
}
