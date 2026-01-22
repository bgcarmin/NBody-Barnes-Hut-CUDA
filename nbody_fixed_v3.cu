#include <cuda_runtime.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <GL/freeglut.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define THREADS_PER_BLOCK 256
#define G_CONST 1.0f
#define THETA 1.0f
#define DT 0.05f
#define SOFTENING 10.0f
#define DAMPING 0.999999f
#define MAX_SPEED 300.0f

struct OctreeNode {
    float mass;
    float comX, comY, comZ;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    int children[8];
    int parent;
};

// Globalne varijable
int N = 15000; // Povećan broj čestica za bolji efekt
float camDistance = 3000.0f;
float camRotX = 30.0f, camRotY = 45.0f;
int lastMouseX, lastMouseY;
bool mouseRotate = false;

float *d_posX, *d_posY, *d_posZ;
float *d_velX, *d_velY, *d_velZ;
float *d_accX, *d_accY, *d_accZ;
float *d_mass;
unsigned int *d_mortonCodes;
int *d_indices;
OctreeNode *d_nodes;
int *d_nodeCounter, *d_leafNodeIdx;
float *d_bounds; 

GLuint vbo_positions, vbo_colors;
cudaGraphicsResource *res_pos, *res_col;

// ============================================================================
// KERNELI
// ============================================================================

__device__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__global__ void computeMortonCodesKernel(const float *posX, const float *posY, const float *posZ, float *bounds, unsigned int *mortonCodes, int *indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float minX = bounds[0], size = fmaxf(bounds[3] - bounds[0], 1.0f);
    unsigned int x = (unsigned int)((posX[i] - minX) / size * 1023.0f);
    unsigned int y = (unsigned int)((posY[i] - bounds[1]) / size * 1023.0f);
    unsigned int z = (unsigned int)((posZ[i] - bounds[2]) / size * 1023.0f);
    mortonCodes[i] = (expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z);
    indices[i] = i;
}

__global__ void buildOctreeKernel(const float *posX, const float *posY, const float *posZ, OctreeNode *nodes, int n, int *nodeCounter, float *bounds, int *leafNodeIdx, const int *indices) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *nodeCounter = 1;
        nodes[0].minX = bounds[0]; nodes[0].minY = bounds[1]; nodes[0].minZ = bounds[2];
        nodes[0].maxX = bounds[3]; nodes[0].maxY = bounds[4]; nodes[0].maxZ = bounds[5];
        nodes[0].mass = 0; nodes[0].parent = -1;
        for(int k=0; k<8; k++) nodes[0].children[k] = -1;
        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            int nIdx = 0, depth = 0;
            while (depth < 25) {
                OctreeNode *node = &nodes[nIdx];
                float mx = (node->minX + node->maxX) * 0.5f, my = (node->minY + node->maxY) * 0.5f, mz = (node->minZ + node->maxZ) * 0.5f;
                int oct = (posX[idx] >= mx) | ((posY[idx] >= my) << 1) | ((posZ[idx] >= mz) << 2);
                int child = node->children[oct];
                if (child == -1) { node->children[oct] = idx; leafNodeIdx[idx] = nIdx; break; }
                else if (child < n) {
                    int oldB = child; int next = atomicAdd(nodeCounter, 1);
                    nodes[next].parent = nIdx; nodes[next].mass = 0; for(int k=0; k<8; k++) nodes[next].children[k] = -1;
                    nodes[next].minX = (oct&1)?mx:node->minX; nodes[next].maxX = (oct&1)?node->maxX:mx;
                    nodes[next].minY = (oct&2)?my:node->minY; nodes[next].maxY = (oct&2)?node->maxY:my;
                    nodes[next].minZ = (oct&4)?mz:node->minZ; nodes[next].maxZ = (oct&4)?node->maxZ:mz;
                    int oldOct = (posX[oldB] >= (nodes[next].minX+nodes[next].maxX)*0.5f) | ((posY[oldB] >= (nodes[next].minY+nodes[next].maxY)*0.5f) << 1) | ((posZ[oldB] >= (nodes[next].minZ+nodes[next].maxZ)*0.5f) << 2);
                    nodes[next].children[oldOct] = oldB; leafNodeIdx[oldB] = next; node->children[oct] = next; nIdx = next;
                } else { nIdx = child; }
                depth++;
            }
        }
    }
}

__global__ void integrateKernel(float *posX, float *posY, float *posZ, float *velX, float *velY, float *velZ, const float *accX, const float *accY, const float *accZ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vx = (velX[i] + accX[i] * DT) * DAMPING;
    float vy = (velY[i] + accY[i] * DT) * DAMPING;
    float vz = (velZ[i] + accZ[i] * DT) * DAMPING;
    float speedSq = vx*vx + vy*vy + vz*vz;
    if (speedSq > MAX_SPEED * MAX_SPEED) { float scale = MAX_SPEED / sqrtf(speedSq); vx *= scale; vy *= scale; vz *= scale; }
    velX[i] = vx; velY[i] = vy; velZ[i] = vz;
    posX[i] += vx * DT; posY[i] += vy * DT; posZ[i] += vz * DT;
}

__global__ void updateVisualsKernel(float *vbo_p, float *vbo_c, const float *posX, const float *posY, const float *posZ, const float *velX, const float *velY, const float *velZ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vbo_p[i*3] = posX[i]; vbo_p[i*3+1] = posY[i]; vbo_p[i*3+2] = posZ[i];
        float speed = sqrtf(velX[i]*velX[i] + velY[i]*velY[i] + velZ[i]*velZ[i]);
        float t = fminf(speed / 100.0f, 1.0f);
        vbo_c[i*3] = 0.3f + t * 0.7f; // Red
        vbo_c[i*3+1] = 0.2f + t * 0.5f; // Green
        vbo_c[i*3+2] = 1.0f - t * 0.5f; // Blue
    }
}

// ... (computeBoundingBox, computeCOM, finalizeCOM, computeForce su isti kao prije)
__global__ void computeBoundingBoxKernel(const float *posX, const float *posY, const float *posZ, int n, float *bounds) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float minX = 1e10f, minY = 1e10f, minZ = 1e10f, maxX = -1e10f, maxY = -1e10f, maxZ = -1e10f;
        for (int i = 0; i < n; i++) {
            minX = fminf(minX, posX[i]); minY = fminf(minY, posY[i]); minZ = fminf(minZ, posZ[i]);
            maxX = fmaxf(maxX, posX[i]); maxY = fmaxf(maxY, posY[i]); maxZ = fmaxf(maxZ, posZ[i]);
        }
        float size = fmaxf(maxX - minX, fmaxf(maxY - minY, maxZ - minZ));
        bounds[0] = minX; bounds[1] = minY; bounds[2] = minZ;
        bounds[3] = minX + size; bounds[4] = minY + size; bounds[5] = minZ + size;
    }
}

__global__ void computeCOMKernel(OctreeNode *nodes, const float *posX, const float *posY, const float *posZ, const float *mass, int n, int *leafNodeIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int curr = leafNodeIdx[i];
    float m = mass[i], cx = posX[i]*m, cy = posY[i]*m, cz = posZ[i]*m;
    while (curr != -1) { atomicAdd(&nodes[curr].mass, m); atomicAdd(&nodes[curr].comX, cx); atomicAdd(&nodes[curr].comY, cy); atomicAdd(&nodes[curr].comZ, cz); curr = nodes[curr].parent; }
}

__global__ void finalizeCOMKernel(OctreeNode *nodes, int nodeCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nodeCount && nodes[i].mass > 1e-6f) { nodes[i].comX /= nodes[i].mass; nodes[i].comY /= nodes[i].mass; nodes[i].comZ /= nodes[i].mass; }
}

__global__ void computeForceKernel(const float *posX, const float *posY, const float *posZ, float *accX, float *accY, float *accZ, const OctreeNode *nodes, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = posX[i], py = posY[i], pz = posZ[i], ax = 0, ay = 0, az = 0;
    int stack[64], ptr = 0; stack[ptr++] = 0;
    while (ptr > 0) {
        int idx = stack[--ptr]; const OctreeNode *node = &nodes[idx];
        if (node->mass <= 0) continue;
        float dx = node->comX - px, dy = node->comY - py, dz = node->comZ - pz;
        float d2 = dx*dx + dy*dy + dz*dz + SOFTENING; float dist = sqrtf(d2);
        if (idx < n || (node->maxX - node->minX) / dist < THETA) { float f = G_CONST * node->mass / (d2 * dist); ax += f * dx; ay += f * dy; az += f * dz; }
        else { for (int k = 0; k < 8; k++) if (node->children[k] != -1) stack[ptr++] = node->children[k]; }
    }
    accX[i] = ax; accY[i] = ay; accZ[i] = az;
}

// ============================================================================
// MAIN & DISPLAY
// ============================================================================

void simulationStep() {
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    computeBoundingBoxKernel<<<1, 1>>>(d_posX, d_posY, d_posZ, N, d_bounds);
    computeMortonCodesKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ, d_bounds, d_mortonCodes, d_indices, N);
    thrust::device_ptr<unsigned int> d_keys(d_mortonCodes); thrust::device_ptr<int> d_vals(d_indices);
    thrust::sort_by_key(d_keys, d_keys + N, d_vals);
    cudaMemset(d_nodes, 0, (N * 2) * sizeof(OctreeNode));
    buildOctreeKernel<<<1, 1>>>(d_posX, d_posY, d_posZ, d_nodes, N, d_nodeCounter, d_bounds, d_leafNodeIdx, d_indices);
    int hCount; cudaMemcpy(&hCount, d_nodeCounter, 4, cudaMemcpyDeviceToHost);
    computeCOMKernel<<<blocks, THREADS_PER_BLOCK>>>(d_nodes, d_posX, d_posY, d_posZ, d_mass, N, d_leafNodeIdx);
    finalizeCOMKernel<<<(hCount+255)/256, 256>>>(d_nodes, hCount);
    computeForceKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ, d_accX, d_accY, d_accZ, d_nodes, N);
    integrateKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, d_accX, d_accY, d_accZ, N);
}

void display() {
    simulationStep();
    float *ptr_p, *ptr_c; size_t b;
    cudaGraphicsMapResources(2, &res_pos, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&ptr_p, &b, res_pos);
    cudaGraphicsResourceGetMappedPointer((void**)&ptr_c, &b, res_col);
    updateVisualsKernel<<<(N+255)/256, 256>>>(ptr_p, ptr_c, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, N);
    cudaGraphicsUnmapResources(2, &res_pos, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity(); gluLookAt(0, 0, camDistance, 0, 0, 0, 0, 1, 0);
    glRotatef(camRotX, 1, 0, 0); glRotatef(camRotY, 0, 1, 0);
    
    glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions); glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors); glColorPointer(3, GL_FLOAT, 0, 0);
    glPointSize(1.5f); glDrawArrays(GL_POINTS, 0, N);
    glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers(); glutPostRedisplay();
}

int main(int argc, char **argv) {
    // Inicijalizacija OpenGL-a
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1280, 720); 
    glutCreateWindow("Barnes-Hut: Formiranje Struktura");
    glewInit();

    // Host memorija za pripremu podataka
    float *h_px = new float[N];
    float *h_py = new float[N];
    float *h_pz = new float[N];
    float *h_vx = new float[N];
    float *h_vy = new float[N];
    float *h_vz = new float[N];
    float *h_m  = new float[N];

    // Seme za nasumične brojeve
    srand(42); 

    for (int i = 0; i < N; i++) {
        float r, a;
        
        // 1. Raspodjela: 15% čestica ide u jezgru (bulge), 85% u disk
        if (i < N * 0.15f) {
            // Gusta jezgra u sredini
            r = ((float)rand() / RAND_MAX) * 150.0f;
            a = ((float)rand() / RAND_MAX) * 2.0f * 3.14159f;
            h_m[i] = 20.0f + ((float)rand() / RAND_MAX) * 30.0f; // Teže čestice u centru
        } else {
            // Eksponencijalni disk: više čestica bliže centru, ali se šire prema van
            // Koristimo inverznu transformaciju za r = -scale * ln(1 - u)
            float u = (float)rand() / RAND_MAX;
            r = 150.0f - 450.0f * logf(1.0f - u * 0.95f); 
            a = ((float)rand() / RAND_MAX) * 2.0f * 3.14159f;
            h_m[i] = 10.0f + ((float)rand() / RAND_MAX) * 15.0f;
        }

        // Pozicije
        h_px[i] = r * cos(a);
        h_py[i] = r * sin(a);
        h_pz[i] = ((float)rand() / RAND_MAX - 0.5f) * (r * 0.12f); // Disk je deblji u sredini

        // 2. Orbitalna brzina v = sqrt(G * M_unutar / r)
        // Aproksimiramo masu unutar radijusa r kako bi dobili stabilnu rotaciju
        float cumulativeMass = 150000.0f * (r / 1000.0f);
        if (r < 150.0f) cumulativeMass = 80000.0f; // Jača sila u jezgri

        float v_mag = sqrtf(G_CONST * cumulativeMass / (r + SOFTENING));
        
        // Dodajemo malu varijaciju (dispersion) od 10% kako bi potaknuli nestabilnost
        // Bez ovog šuma, sve bi bilo presavršeno i ne bi se formirale grupe
        float v_noise = 0.9f + ((float)rand() / RAND_MAX) * 0.2f;
        
        h_vx[i] = -sin(a) * v_mag * v_noise;
        h_vy[i] =  cos(a) * v_mag * v_noise;
        h_vz[i] = ((float)rand() / RAND_MAX - 0.5f) * 5.0f;
    }

    // GPU Alokacija
    cudaMalloc(&d_posX, N * 4); cudaMalloc(&d_posY, N * 4); cudaMalloc(&d_posZ, N * 4);
    cudaMalloc(&d_velX, N * 4); cudaMalloc(&d_velY, N * 4); cudaMalloc(&d_velZ, N * 4);
    cudaMalloc(&d_accX, N * 4); cudaMalloc(&d_accY, N * 4); cudaMalloc(&d_accZ, N * 4);
    cudaMalloc(&d_mass, N * 4); 
    
    cudaMalloc(&d_nodes, (N * 2) * sizeof(OctreeNode));
    cudaMalloc(&d_nodeCounter, 4); 
    cudaMalloc(&d_leafNodeIdx, N * 4); 
    cudaMalloc(&d_bounds, 24);
    cudaMalloc(&d_mortonCodes, N * 4); 
    cudaMalloc(&d_indices, N * 4);

    // Kopiranje podataka na Device
    cudaMemcpy(d_posX, h_px, N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_py, N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, h_pz, N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX, h_vx, N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, h_vy, N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, h_vz, N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_m,  N * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_accX, 0, N * 4);
    cudaMemset(d_accY, 0, N * 4);
    cudaMemset(d_accZ, 0, N * 4);

    // Oslobađanje host memorije
    delete[] h_px; delete[] h_py; delete[] h_pz;
    delete[] h_vx; delete[] h_vy; delete[] h_vz; delete[] h_m;

    // OpenGL VBO Setup
    glGenBuffers(1, &vbo_positions); 
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions); 
    glBufferData(GL_ARRAY_BUFFER, N * 3 * 4, 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&res_pos, vbo_positions, cudaGraphicsMapFlagsWriteDiscard);

    glGenBuffers(1, &vbo_colors); 
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors); 
    glBufferData(GL_ARRAY_BUFFER, N * 3 * 4, 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&res_col, vbo_colors, cudaGraphicsMapFlagsWriteDiscard);

    // Callbacks
    glutDisplayFunc(display);
    glutMouseFunc([](int b, int s, int x, int y){
        if(b==0) { mouseRotate=(s==0); lastMouseX=x; lastMouseY=y; }
        if(b==3) camDistance -= 200.0f;
        if(b==4) camDistance += 200.0f;
    });
    glutMotionFunc([](int x, int y){
        if(mouseRotate){
            camRotY += (x - lastMouseX) * 0.2f;
            camRotX += (y - lastMouseY) * 0.2f;
            lastMouseX = x; lastMouseY = y;
        }
    });
    glutReshapeFunc([](int w, int h){
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, (float)w/h, 10.0f, 500000.0f);
        glMatrixMode(GL_MODELVIEW);
    });

    glEnable(GL_DEPTH_TEST);
    glutMainLoop();
    return 0;
}