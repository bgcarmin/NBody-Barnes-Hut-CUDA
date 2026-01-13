// nbody_cuda_no_viz.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================

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
    int *bodyIndices,
    int numBodies,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies) return;
    
    mortonCodes[idx] = morton3D(posX[idx], posY[idx], posZ[idx],
                                 minX, minY, minZ, maxX, maxY, maxZ);
    bodyIndices[idx] = idx;
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

__global__ void initOctreeKernel(
    OctreeNode *nodes,
    int maxNodes,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < maxNodes) {
        nodes[idx].mass = -1.0f;  // Negative = not computed
        nodes[idx].bodyIndex = -1;
        nodes[idx].numBodies = 0;
        for (int i = 0; i < 8; i++) {
            nodes[idx].children[i] = -1;
        }
    }
    
    // Initialize root node
    if (idx == 0) {
        nodes[0].minX = minX;
        nodes[0].minY = minY;
        nodes[0].minZ = minZ;
        nodes[0].maxX = maxX;
        nodes[0].maxY = maxY;
        nodes[0].maxZ = maxZ;
    }
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
        int depth = 0;
        const int MAX_DEPTH = 20;
        
        while (!inserted && depth < MAX_DEPTH) {
            OctreeNode *node = &nodes[nodeIdx];
            
            float midX = (node->minX + node->maxX) * 0.5f;
            float midY = (node->minY + node->maxY) * 0.5f;
            float midZ = (node->minZ + node->maxZ) * 0.5f;
            
            int octant = getOctant(px, py, pz, midX, midY, midZ);
            int childIdx = node->children[octant];
            
            if (childIdx != LOCKED) {
                int old = atomicCAS(&node->children[octant], childIdx, LOCKED);
                
                if (old == childIdx) {
                    if (old == -1) {
                        // Empty slot - insert body
                        node->children[octant] = bodyIdx;
                        inserted = true;
                    } else if (old < numBodies) {
                        // Slot contains body - need to subdivide
                        int newNodeIdx = atomicAdd(nodeCounter, 1);
                        
                        if (newNodeIdx < numBodies * 8) {  // Safety check
                            OctreeNode *newNode = &nodes[newNodeIdx + numBodies];
                            
                            newNode->minX = (octant & 1) ? midX : node->minX;
                            newNode->maxX = (octant & 1) ? node->maxX : midX;
                            newNode->minY = (octant & 2) ? midY : node->minY;
                            newNode->maxY = (octant & 2) ? node->maxY : midY;
                            newNode->minZ = (octant & 4) ? midZ : node->minZ;
                            newNode->maxZ = (octant & 4) ? node->maxZ : midZ;
                            newNode->bodyIndex = -1;
                            newNode->mass = -1.0f;
                            for (int k = 0; k < 8; k++) newNode->children[k] = -1;
                            
                            // Insert old body into new node
                            float oldPx = posX[old];
                            float oldPy = posY[old];
                            float oldPz = posZ[old];
                            float newMidX = (newNode->minX + newNode->maxX) * 0.5f;
                            float newMidY = (newNode->minY + newNode->maxY) * 0.5f;
                            float newMidZ = (newNode->minZ + newNode->maxZ) * 0.5f;
                            int oldOctant = getOctant(oldPx, oldPy, oldPz, newMidX, newMidY, newMidZ);
                            newNode->children[oldOctant] = old;
                            
                            __threadfence();
                            
                            // Attach new node
                            node->children[octant] = newNodeIdx + numBodies;
                            
                            // Continue insertion with new node
                            nodeIdx = newNodeIdx + numBodies;
                            depth++;
                        } else {
                            inserted = true;  // Overflow protection
                        }
                    } else {
                        // Slot contains internal node - descend
                        nodeIdx = old;
                        depth++;
                    }
                } else {
                    // Lock failed - retry with throttling
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
    for (int i = totalNodes - 1 - idx; i >= 0; i -= stride) {
        if (i < numBodies) continue;  // Skip body slots
        
        OctreeNode *node = &nodes[i];
        
        float totalMass = 0.0f;
        float comX = 0.0f, comY = 0.0f, comZ = 0.0f;
        int childCount = 0;
        
        __shared__ int missingChildren[THREADS_PER_BLOCK];
        int missing = 0;
        
        bool success = false;
        int retries = 0;
        const int MAX_RETRIES = 100;
        
        while (!success && retries < MAX_RETRIES) {
            missing = 0;
            totalMass = 0.0f;
            comX = comY = comZ = 0.0f;
            childCount = 0;
            
            // Check all children
            for (int k = 0; k < 8; k++) {
                int childIdx = node->children[k];
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
                        // Child is a node - check if ready
                        OctreeNode *child = &nodes[childIdx];
                        if (child->mass >= 0.0f) {
                            float m = child->mass;
                            totalMass += m;
                            comX += child->comX * m;
                            comY += child->comY * m;
                            comZ += child->comZ * m;
                            childCount++;
                        } else {
                            missing++;
                        }
                    }
                }
            }
            
            if (missing == 0) {
                success = true;
            } else {
                retries++;
                __syncthreads();  // Throttle
            }
        }
        
        // Finalize center of mass
        if (totalMass > 0.0f) {
            float invMass = 1.0f / totalMass;
            node->comX = comX * invMass;
            node->comY = comY * invMass;
            node->comZ = comZ * invMass;
        } else {
            node->comX = (node->minX + node->maxX) * 0.5f;
            node->comY = (node->minY + node->maxY) * 0.5f;
            node->comZ = (node->minZ + node->maxZ) * 0.5f;
        }
        
        node->numBodies = childCount;
        __threadfence();
        node->mass = totalMass;  // Release "ready flag"
    }
}

// ============================================================================
// KERNEL 4: SORT BODIES (optional but improves performance)
// ============================================================================

__global__ void moveChildrenToFrontKernel(
    OctreeNode *nodes,
    int numBodies,
    int totalNodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx + numBodies; i < totalNodes; i += stride) {
        OctreeNode *node = &nodes[i];
        
        int validChildren[8];
        int count = 0;
        
        for (int k = 0; k < 8; k++) {
            if (node->children[k] >= 0) {
                validChildren[count++] = node->children[k];
            }
        }
        
        for (int k = 0; k < 8; k++) {
            node->children[k] = (k < count) ? validChildren[k] : -1;
        }
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
        const int MAX_DEPTH = 63;
        
        while (depth >= 0 && depth < MAX_DEPTH) {
            // First thread in warp loads node data
            if (laneId == 0) {
                int nodeIdx = sStack[warpId][depth];
                const OctreeNode *node = &nodes[nodeIdx];
                
                sNodeData[warpId][0] = node->mass;
                sNodeData[warpId][1] = node->comX;
                sNodeData[warpId][2] = node->comY;
                sNodeData[warpId][3] = node->comZ;
                sNodeData[warpId][4] = node->maxX - node->minX;  // width
                sNodeData[warpId][5] = (float)node->bodyIndex;
            }
            __threadfence_block();
            
            float nodeMass = sNodeData[warpId][0];
            float nodeComX = sNodeData[warpId][1];
            float nodeComY = sNodeData[warpId][2];
            float nodeComZ = sNodeData[warpId][3];
            float nodeWidth = sNodeData[warpId][4];
            int nodeBodyIdx = (int)sNodeData[warpId][5];
            
            if (nodeMass > 0.0f && nodeBodyIdx != bodyIdx) {
                float dx = nodeComX - px;
                float dy = nodeComY - py;
                float dz = nodeComZ - pz;
                float distSq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
                float dist = sqrtf(distSq);
                
                // Barnes-Hut criterion: width / distance < theta
                bool farEnough = (nodeWidth / (dist + EPSILON)) < theta;
                
                // Use __ballot_sync for warp voting (CUDA 9.0+)
                unsigned int mask = __ballot_sync(0xFFFFFFFF, farEnough);
                
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
                        const OctreeNode *node = &nodes[nodeIdx];
                        
                        int childCount = 0;
                        for (int k = 0; k < 8; k++) {
                            if (node->children[k] >= 0) {
                                sStack[warpId][depth + childCount] = node->children[k];
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
// ERROR CHECKING
// ============================================================================

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        
        // Dodatne informacije
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        fprintf(stderr, "Number of CUDA devices: %d\n", deviceCount);
        
        if (deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            fprintf(stderr, "Device 0: %s\n", prop.name);
            fprintf(stderr, "Compute Capability: %d.%d\n", prop.major, prop.minor);
        }
        
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// MAIN SIMULATION
// ============================================================================

int main() {
    // Inicijalizacija CUDA i provera dostupnosti
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "CUDA Error: No CUDA-capable device detected!\n");
        fprintf(stderr, "Error code: %d - %s\n", error, cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    
    // Postavi device i prikaži informacije
    cudaSetDevice(0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================\n");
    printf("CUDA Barnes-Hut N-Body Simulation\n");
    printf("========================================\n");
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("========================================\n");
    
    const int N = 500;
    const int NUM_STEPS = 10;
    const int NUM_BLOCKS = 256;
    
    printf("Number of bodies: %d\n", N);
    printf("Number of time steps: %d\n", NUM_STEPS);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Number of blocks: %d\n", NUM_BLOCKS);
    printf("========================================\n\n");
    
    // Allocate host memory
    float *h_posX = new float[N];
    float *h_posY = new float[N];
    float *h_posZ = new float[N];
    float *h_velX = new float[N];
    float *h_velY = new float[N];
    float *h_velZ = new float[N];
    float *h_mass = new float[N];
    
    // Initialize bodies (Plummer sphere)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    printf("Initializing bodies...\n");
    for (int i = 0; i < N; i++) {
        float angle = dist(rng) * 2.0f * (float)M_PI;
        float r = dist(rng) * 400.0f;
        h_posX[i] = r * cosf(angle);
        h_posY[i] = r * sinf(angle);
        h_posZ[i] = (dist(rng) - 0.5f) * 100.0f;
        
        h_mass[i] = 1000.0f;
        
        float v = sqrtf(G_CONST * 1000.0f * N / (r + 50.0f));
        h_velX[i] = -h_posY[i] * 0.01f;
        h_velY[i] = h_posX[i] * 0.01f;
        h_velZ[i] = 0.0f;
    }
    
    // Allocate device memory
    float *d_posX, *d_posY, *d_posZ;
    float *d_velX, *d_velY, *d_velZ;
    float *d_accX, *d_accY, *d_accZ;
    float *d_mass;
    
    printf("Allocating GPU memory...\n");
    checkCudaError(cudaMalloc(&d_posX, N * sizeof(float)), "cudaMalloc posX");
    checkCudaError(cudaMalloc(&d_posY, N * sizeof(float)), "cudaMalloc posY");
    checkCudaError(cudaMalloc(&d_posZ, N * sizeof(float)), "cudaMalloc posZ");
    checkCudaError(cudaMalloc(&d_velX, N * sizeof(float)), "cudaMalloc velX");
    checkCudaError(cudaMalloc(&d_velY, N * sizeof(float)), "cudaMalloc velY");
    checkCudaError(cudaMalloc(&d_velZ, N * sizeof(float)), "cudaMalloc velZ");
    checkCudaError(cudaMalloc(&d_accX, N * sizeof(float)), "cudaMalloc accX");
    checkCudaError(cudaMalloc(&d_accY, N * sizeof(float)), "cudaMalloc accY");
    checkCudaError(cudaMalloc(&d_accZ, N * sizeof(float)), "cudaMalloc accZ");
    checkCudaError(cudaMalloc(&d_mass, N * sizeof(float)), "cudaMalloc mass");
    
    // Copy data to device
    printf("Copying data to GPU...\n");
    checkCudaError(cudaMemcpy(d_posX, h_posX, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy posX");
    checkCudaError(cudaMemcpy(d_posY, h_posY, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy posY");
    checkCudaError(cudaMemcpy(d_posZ, h_posZ, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy posZ");
    checkCudaError(cudaMemcpy(d_velX, h_velX, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy velX");
    checkCudaError(cudaMemcpy(d_velY, h_velY, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy velY");
    checkCudaError(cudaMemcpy(d_velZ, h_velZ, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy velZ");
    checkCudaError(cudaMemcpy(d_mass, h_mass, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy mass");
    
    // Allocate octree memory
    const int MAX_NODES = N * 8;
    OctreeNode *d_nodes;
    checkCudaError(cudaMalloc(&d_nodes, MAX_NODES * sizeof(OctreeNode)), "cudaMalloc nodes");
    
    // Bounding box arrays
    float *d_blockMinX, *d_blockMinY, *d_blockMinZ;
    float *d_blockMaxX, *d_blockMaxY, *d_blockMaxZ;
    int *d_counter;
    
    checkCudaError(cudaMalloc(&d_blockMinX, NUM_BLOCKS * sizeof(float)), "cudaMalloc blockMinX");
    checkCudaError(cudaMalloc(&d_blockMinY, NUM_BLOCKS * sizeof(float)), "cudaMalloc blockMinY");
    checkCudaError(cudaMalloc(&d_blockMinZ, NUM_BLOCKS * sizeof(float)), "cudaMalloc blockMinZ");
    checkCudaError(cudaMalloc(&d_blockMaxX, NUM_BLOCKS * sizeof(float)), "cudaMalloc blockMaxX");
    checkCudaError(cudaMalloc(&d_blockMaxY, NUM_BLOCKS * sizeof(float)), "cudaMalloc blockMaxY");
    checkCudaError(cudaMalloc(&d_blockMaxZ, NUM_BLOCKS * sizeof(float)), "cudaMalloc blockMaxZ");
    checkCudaError(cudaMalloc(&d_counter, sizeof(int)), "cudaMalloc counter");
    
    // Morton code arrays (for sorting)
    unsigned long long *d_mortonCodes;
    int *d_sortedIndices;
    checkCudaError(cudaMalloc(&d_mortonCodes, N * sizeof(unsigned long long)), "cudaMalloc morton");
    checkCudaError(cudaMalloc(&d_sortedIndices, N * sizeof(int)), "cudaMalloc indices");
    
    // Node counter
    int *d_nodeCounter;
    checkCudaError(cudaMalloc(&d_nodeCounter, sizeof(int)), "cudaMalloc nodeCounter");
    
    printf("\nStarting simulation...\n\n");
    
    // Main simulation loop
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < NUM_STEPS; step++) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        
        // Reset counter
        checkCudaError(cudaMemset(d_counter, 0, sizeof(int)), "memset counter");
        checkCudaError(cudaMemset(d_nodeCounter, 0, sizeof(int)), "memset nodeCounter");
        
        // KERNEL 1: Compute bounding box
        computeBoundingBoxKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_posX, d_posY, d_posZ, N,
            d_blockMinX, d_blockMinY, d_blockMinZ,
            d_blockMaxX, d_blockMaxY, d_blockMaxZ,
            d_counter
        );
        checkCudaError(cudaDeviceSynchronize(), "Kernel 1");
        
        // Copy bounding box back
        float h_minX, h_minY, h_minZ, h_maxX, h_maxY, h_maxZ;
        checkCudaError(cudaMemcpy(&h_minX, d_blockMinX, sizeof(float), cudaMemcpyDeviceToHost), "copy minX");
        checkCudaError(cudaMemcpy(&h_minY, d_blockMinY, sizeof(float), cudaMemcpyDeviceToHost), "copy minY");
        checkCudaError(cudaMemcpy(&h_minZ, d_blockMinZ, sizeof(float), cudaMemcpyDeviceToHost), "copy minZ");
        checkCudaError(cudaMemcpy(&h_maxX, d_blockMaxX, sizeof(float), cudaMemcpyDeviceToHost), "copy maxX");
        checkCudaError(cudaMemcpy(&h_maxY, d_blockMaxY, sizeof(float), cudaMemcpyDeviceToHost), "copy maxY");
        checkCudaError(cudaMemcpy(&h_maxZ, d_blockMaxZ, sizeof(float), cudaMemcpyDeviceToHost), "copy maxZ");
        
        // KERNEL 2: Initialize octree
        initOctreeKernel<<<(MAX_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_nodes, MAX_NODES, h_minX, h_minY, h_minZ, h_maxX, h_maxY, h_maxZ
        );
        checkCudaError(cudaDeviceSynchronize(), "Init Octree");
        
        // KERNEL 2: Build octree
        buildOctreeKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_posX, d_posY, d_posZ, d_mass, nullptr,
            d_nodes, N, d_nodeCounter,
            h_minX, h_minY, h_minZ, h_maxX, h_maxY, h_maxZ
        );
        checkCudaError(cudaDeviceSynchronize(), "Kernel 2");
        
        // Get total nodes
        int h_nodeCounter;
        checkCudaError(cudaMemcpy(&h_nodeCounter, d_nodeCounter, sizeof(int), cudaMemcpyDeviceToHost), "copy nodeCounter");
        int totalNodes = N + h_nodeCounter;
        
        // KERNEL 3: Compute center of mass
        computeCenterOfMassKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_nodes, d_posX, d_posY, d_posZ, d_mass, N, totalNodes
        );
        checkCudaError(cudaDeviceSynchronize(), "Kernel 3");
        
        // KERNEL 4: Move children to front (optimization)
        moveChildrenToFrontKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_nodes, N, totalNodes
        );
        checkCudaError(cudaDeviceSynchronize(), "Kernel 4");
        
        // KERNEL 5: Compute forces
        computeForceKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_posX, d_posY, d_posZ, d_mass, nullptr,
            d_accX, d_accY, d_accZ, d_nodes, N, THETA
        );
        checkCudaError(cudaDeviceSynchronize(), "Kernel 5");
        
        // KERNEL 6: Integrate
        integrateKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_posX, d_posY, d_posZ,
            d_velX, d_velY, d_velZ,
            d_accX, d_accY, d_accZ,
            N, DT
        );
        checkCudaError(cudaDeviceSynchronize(), "Kernel 6");
        
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> stepTime = stepEnd - stepStart;
        
        if (step % 10 == 0) {
            printf("Step %4d/%d | Time: %8.2f ms | Nodes: %d\n", 
                   step, NUM_STEPS, stepTime.count(), totalNodes);
        }
    }
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = totalEnd - totalStart;
    
    printf("\n========================================\n");
    printf("Simulation completed!\n");
    printf("Total time: %.2f seconds\n", totalTime.count());
    printf("Average time per step: %.2f ms\n", totalTime.count() * 1000.0 / NUM_STEPS);
    printf("========================================\n");
    
    // Copy final results back
    printf("\nCopying results back to host...\n");
    checkCudaError(cudaMemcpy(h_posX, d_posX, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posX");
    checkCudaError(cudaMemcpy(h_posY, d_posY, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posY");
    checkCudaError(cudaMemcpy(h_posZ, d_posZ, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posZ");
    
    // Save to file
    FILE *fp = fopen("output.txt", "w");
    if (fp) {
        fprintf(fp, "# Final positions after %d steps\n", NUM_STEPS);
        fprintf(fp, "# Format: x y z\n");
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6f %.6f %.6f\n", h_posX[i], h_posY[i], h_posZ[i]);
        }
        fclose(fp);
        printf("Results saved to output.txt\n");
    }
    
    // Cleanup
    cudaFree(d_posX);
    cudaFree(d_posY);
    cudaFree(d_posZ);
    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velZ);
    cudaFree(d_accX);
    cudaFree(d_accY);
    cudaFree(d_accZ);
    cudaFree(d_mass);
    cudaFree(d_nodes);
    cudaFree(d_blockMinX);
    cudaFree(d_blockMinY);
    cudaFree(d_blockMinZ);
    cudaFree(d_blockMaxX);
    cudaFree(d_blockMaxY);
    cudaFree(d_blockMaxZ);
    cudaFree(d_counter);
    cudaFree(d_mortonCodes);
    cudaFree(d_sortedIndices);
    cudaFree(d_nodeCounter);
    
    delete[] h_posX;
    delete[] h_posY;
    delete[] h_posZ;
    delete[] h_velX;
    delete[] h_velY;
    delete[] h_velZ;
    delete[] h_mass;
    
    printf("\nDone!\n");
    return 0;
}
