// nbody_cuda_barneshut.cu - Stabilan Barnes-Hut sa bottom-up octree
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define G_CONST 1.0f
#define THETA 0.5f
#define DT 0.03f
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
// KERNEL 2: SIMPLE OCTREE (bez lock-a, sekvencijalno po Morton kodu)
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
    // Samo jedan thread gradi drvo (jednostavno, ali radi)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *nodeCounter = 0;
        
        // Initialize root
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
        
        // Insert bodies sequentially
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
                    // Empty - insert body
                    node->children[octant] = bodyIdx;
                    break;
                } else if (childIdx < numBodies) {
                    // Body already here - create new node
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
                    
                    // Move old body to new node
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
                    // Internal node - descend
                    nodeIdx = childIdx;
                }
                
                depth++;
            }
        }
    }
}

// ============================================================================
// KERNEL 3: CENTER OF MASS (parallelno bottom-up)
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
    
    // Stack-based traversal
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
        
        // Check if this is a body node pointing to current body
        bool isSelf = (nodeIdx < numBodies && nodeIdx == i);
        
        if ((farEnough || nodeIdx < numBodies) && !isSelf) {
            // Use approximation
            float invDist = rsqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;
            float f = G_CONST * node->mass * invDist3;
            
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        } else if (!isSelf) {
            // Descend to children
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

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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
// MAIN
// ============================================================================

int main() {
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
    printf("Barnes-Hut N-Body Simulation (CUDA)\n");
    printf("========================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("========================================\n");
    
    const int N = 10000;
    const int NUM_STEPS = 100;
    
    printf("Bodies: %d\n", N);
    printf("Steps: %d\n", NUM_STEPS);
    printf("Theta: %.2f\n", THETA);
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
    float *d_posX, *d_posY, *d_posZ;
    float *d_velX, *d_velY, *d_velZ;
    float *d_accX, *d_accY, *d_accZ;
    float *d_mass;
    
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
    OctreeNode *d_nodes;
    checkCudaError(cudaMalloc(&d_nodes, MAX_NODES * sizeof(OctreeNode)), "malloc nodes");
    
    // Bounding box
    float *d_minX, *d_minY, *d_minZ;
    float *d_maxX, *d_maxY, *d_maxZ;
    checkCudaError(cudaMalloc(&d_minX, sizeof(float)), "malloc minX");
    checkCudaError(cudaMalloc(&d_minY, sizeof(float)), "malloc minY");
    checkCudaError(cudaMalloc(&d_minZ, sizeof(float)), "malloc minZ");
    checkCudaError(cudaMalloc(&d_maxX, sizeof(float)), "malloc maxX");
    checkCudaError(cudaMalloc(&d_maxY, sizeof(float)), "malloc maxY");
    checkCudaError(cudaMalloc(&d_maxZ, sizeof(float)), "malloc maxZ");
    
    // Morton sorting
    unsigned long long *d_morton;
    int *d_indices, *d_nodeCounter;
    checkCudaError(cudaMalloc(&d_morton, N * sizeof(unsigned long long)), "malloc morton");
    checkCudaError(cudaMalloc(&d_indices, N * sizeof(int)), "malloc indices");
    checkCudaError(cudaMalloc(&d_nodeCounter, sizeof(int)), "malloc counter");
    
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("\nStarting simulation...\n\n");
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < NUM_STEPS; step++) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        
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
        
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> stepTime = stepEnd - stepStart;
        
        if (step % 10 == 0) {
            printf("Step %4d/%d | Time: %7.2f ms | Nodes: %d\n",
                   step, NUM_STEPS, stepTime.count(), totalNodes);
        }
    }
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = totalEnd - totalStart;
    
    printf("\n========================================\n");
    printf("Completed!\n");
    printf("Total time: %.2f s\n", totalTime.count());
    printf("Avg time/step: %.2f ms\n", totalTime.count() * 1000.0 / NUM_STEPS);
    printf("========================================\n");
    
    // Copy final results back to host
    printf("\nCopying results back to host...\n");
    checkCudaError(cudaMemcpy(h_posX, d_posX, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posX");
    checkCudaError(cudaMemcpy(h_posY, d_posY, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posY");
    checkCudaError(cudaMemcpy(h_posZ, d_posZ, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posZ");
    checkCudaError(cudaMemcpy(h_velX, d_velX, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy velX");
    checkCudaError(cudaMemcpy(h_velY, d_velY, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy velY");
    checkCudaError(cudaMemcpy(h_velZ, d_velZ, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy velZ");
    
    // Calculate statistics
    float totalKE = 0.0f;
    float minX = h_posX[0], maxX = h_posX[0];
    float minY = h_posY[0], maxY = h_posY[0];
    float minZ = h_posZ[0], maxZ = h_posZ[0];
    
    for (int i = 0; i < N; i++) {
        float vSq = h_velX[i]*h_velX[i] + h_velY[i]*h_velY[i] + h_velZ[i]*h_velZ[i];
        totalKE += 0.5f * h_mass[i] * vSq;
        
        minX = fminf(minX, h_posX[i]);
        maxX = fmaxf(maxX, h_posX[i]);
        minY = fminf(minY, h_posY[i]);
        maxY = fmaxf(maxY, h_posY[i]);
        minZ = fminf(minZ, h_posZ[i]);
        maxZ = fmaxf(maxZ, h_posZ[i]);
    }
    
    printf("\nFinal statistics:\n");
    printf("  Total kinetic energy: %.2e\n", totalKE);
    printf("  Bounding box X: [%.2f, %.2f]\n", minX, maxX);
    printf("  Bounding box Y: [%.2f, %.2f]\n", minY, maxY);
    printf("  Bounding box Z: [%.2f, %.2f]\n", minZ, maxZ);
    
    // Save to file
    printf("\nSaving results to output_bh.txt...\n");
    FILE *fp = fopen("output_bh.txt", "w");
    if (fp) {
        fprintf(fp, "# Barnes-Hut N-Body Simulation Results\n");
        fprintf(fp, "# Final positions and velocities after %d steps\n", NUM_STEPS);
        fprintf(fp, "# Bodies: %d, Theta: %.2f, dt: %.3f\n", N, THETA, DT);
        fprintf(fp, "# Format: x y z vx vy vz\n");
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f\n", 
                    h_posX[i], h_posY[i], h_posZ[i],
                    h_velX[i], h_velY[i], h_velZ[i]);
        }
        fclose(fp);
        printf("Results saved successfully!\n");
    } else {
        fprintf(stderr, "Warning: Could not open output_bh.txt for writing\n");
    }
    
    // Cleanup
    printf("\nCleaning up...\n");
    cudaFree(d_posX); cudaFree(d_posY); cudaFree(d_posZ);
    cudaFree(d_velX); cudaFree(d_velY); cudaFree(d_velZ);
    cudaFree(d_accX); cudaFree(d_accY); cudaFree(d_accZ);
    cudaFree(d_mass); cudaFree(d_nodes);
    cudaFree(d_minX); cudaFree(d_minY); cudaFree(d_minZ);
    cudaFree(d_maxX); cudaFree(d_maxY); cudaFree(d_maxZ);
    cudaFree(d_morton); cudaFree(d_indices); cudaFree(d_nodeCounter);
    
    delete[] h_posX; delete[] h_posY; delete[] h_posZ;
    delete[] h_velX; delete[] h_velY; delete[] h_velZ;
    delete[] h_mass;
    
    printf("\nDone!\n");
    return 0;
}
