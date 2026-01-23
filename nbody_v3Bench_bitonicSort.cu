#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define THREADS_PER_BLOCK 256
#define G_CONST 1.0f
#define THETA 0.5f
#define DT 0.01f
#define SOFTENING 10.0f
#define DAMPING 0.999f
#define MAX_SPEED 500.0f

struct OctreeNode
{
    float mass;
    float comX, comY, comZ;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    int children[8];
    int parent;
};

// ============================================================================
// BITONIC SORT KERNEL
// ============================================================================
__global__ void bitonicSortKernel(unsigned int *keys, int *indices, int j, int k)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i)
    {
        if ((i & k) == 0)
        {
            if (keys[i] > keys[ixj])
            {
                unsigned int tempK = keys[i];
                keys[i] = keys[ixj];
                keys[ixj] = tempK;
                int tempI = indices[i];
                indices[i] = indices[ixj];
                indices[ixj] = tempI;
            }
        }
        else
        {
            if (keys[i] < keys[ixj])
            {
                unsigned int tempK = keys[i];
                keys[i] = keys[ixj];
                keys[ixj] = tempK;
                int tempI = indices[i];
                indices[i] = indices[ixj];
                indices[ixj] = tempI;
            }
        }
    }
}

// ============================================================================
// KERNELI
// ============================================================================

__device__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__global__ void computeBoundingBoxKernel(const float *posX, const float *posY, const float *posZ, int n, float *bounds)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float minX = 1e10f, minY = 1e10f, minZ = 1e10f;
        float maxX = -1e10f, maxY = -1e10f, maxZ = -1e10f;
        for (int i = 0; i < n; i++)
        {
            minX = fminf(minX, posX[i]);
            minY = fminf(minY, posY[i]);
            minZ = fminf(minZ, posZ[i]);
            maxX = fmaxf(maxX, posX[i]);
            maxY = fmaxf(maxY, posY[i]);
            maxZ = fmaxf(maxZ, posZ[i]);
        }
        float size = fmaxf(maxX - minX, fmaxf(maxY - minY, maxZ - minZ));
        bounds[0] = minX;
        bounds[1] = minY;
        bounds[2] = minZ;
        bounds[3] = minX + size;
        bounds[4] = minY + size;
        bounds[5] = minZ + size;
    }
}

__global__ void computeMortonCodesKernel(const float *posX, const float *posY, const float *posZ, float *bounds, unsigned int *mortonCodes, int *indices, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float minX = bounds[0], minY = bounds[1], minZ = bounds[2];
    float size = fmaxf(bounds[3] - bounds[0], 1e-5f);
    unsigned int x = (unsigned int)((posX[i] - minX) / size * 1023.0f);
    unsigned int y = (unsigned int)((posY[i] - minY) / size * 1023.0f);
    unsigned int z = (unsigned int)((posZ[i] - minZ) / size * 1023.0f);
    mortonCodes[i] = (expandBits(x) << 2) | (expandBits(y) << 1) | expandBits(z);
    indices[i] = i;
}

__global__ void buildOctreeKernel(const float *posX, const float *posY, const float *posZ, OctreeNode *nodes, int n, int *nodeCounter, float *bounds, int *leafNodeIdx, const int *indices)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *nodeCounter = 1;
        nodes[0].minX = bounds[0];
        nodes[0].minY = bounds[1];
        nodes[0].minZ = bounds[2];
        nodes[0].maxX = bounds[3];
        nodes[0].maxY = bounds[4];
        nodes[0].maxZ = bounds[5];
        nodes[0].mass = 0;
        nodes[0].parent = -1;
        for (int k = 0; k < 8; k++)
            nodes[0].children[k] = -1;

        for (int i = 0; i < n; i++)
        {
            int idx = indices[i];
            if (idx >= n)
                continue; // Preskoči padding
            int nIdx = 0, depth = 0;
            while (depth < 20)
            {
                OctreeNode *node = &nodes[nIdx];
                float mx = (node->minX + node->maxX) * 0.5f;
                float my = (node->minY + node->maxY) * 0.5f;
                float mz = (node->minZ + node->maxZ) * 0.5f;
                int oct = (posX[idx] >= mx) | ((posY[idx] >= my) << 1) | ((posZ[idx] >= mz) << 2);
                int child = node->children[oct];

                if (child == -1)
                {
                    node->children[oct] = idx;
                    leafNodeIdx[idx] = nIdx;
                    break;
                }
                else if (child < n)
                {
                    int oldB = child;
                    int next = atomicAdd(nodeCounter, 1);
                    nodes[next].parent = nIdx;
                    nodes[next].mass = 0;
                    for (int k = 0; k < 8; k++)
                        nodes[next].children[k] = -1;
                    nodes[next].minX = (oct & 1) ? mx : node->minX;
                    nodes[next].maxX = (oct & 1) ? node->maxX : mx;
                    nodes[next].minY = (oct & 2) ? my : node->minY;
                    nodes[next].maxY = (oct & 2) ? node->maxY : my;
                    nodes[next].minZ = (oct & 4) ? mz : node->minZ;
                    nodes[next].maxZ = (oct & 4) ? node->maxZ : mz;

                    float n_mx = (nodes[next].minX + nodes[next].maxX) * 0.5f;
                    float n_my = (nodes[next].minY + nodes[next].maxY) * 0.5f;
                    float n_mz = (nodes[next].minZ + nodes[next].maxZ) * 0.5f;
                    int oldOct = (posX[oldB] >= n_mx) | ((posY[oldB] >= n_my) << 1) | ((posZ[oldB] >= n_mz) << 2);
                    nodes[next].children[oldOct] = oldB;
                    leafNodeIdx[oldB] = next;
                    node->children[oct] = next;
                    nIdx = next;
                }
                else
                {
                    nIdx = child;
                }
                depth++;
            }
        }
    }
}

__global__ void computeCOMKernel(OctreeNode *nodes, const float *posX, const float *posY, const float *posZ, const float *mass, int n, int *leafNodeIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int curr = leafNodeIdx[i];
    float m = mass[i], cx = posX[i] * m, cy = posY[i] * m, cz = posZ[i] * m;
    while (curr != -1)
    {
        atomicAdd(&nodes[curr].mass, m);
        atomicAdd(&nodes[curr].comX, cx);
        atomicAdd(&nodes[curr].comY, cy);
        atomicAdd(&nodes[curr].comZ, cz);
        curr = nodes[curr].parent;
    }
}

__global__ void finalizeCOMKernel(OctreeNode *nodes, int nodeCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nodeCount)
    {
        float m = nodes[i].mass;
        if (m > 1e-6f)
        {
            float invMass = 1.0f / m;
            nodes[i].comX *= invMass;
            nodes[i].comY *= invMass;
            nodes[i].comZ *= invMass;
        }
    }
}

__global__ void computeForceKernel(const float *posX, const float *posY, const float *posZ, float *accX, float *accY, float *accZ, const OctreeNode *nodes, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float px = posX[i], py = posY[i], pz = posZ[i];
    float ax = 0, ay = 0, az = 0;
    int stack[64];
    int ptr = 0;
    stack[ptr++] = 0;

    while (ptr > 0)
    {
        int idx = stack[--ptr];
        const OctreeNode *node = &nodes[idx];
        if (node->mass <= 0)
            continue;

        float dx = node->comX - px, dy = node->comY - py, dz = node->comZ - pz;
        float d2 = dx * dx + dy * dy + dz * dz + SOFTENING;
        float dist = sqrtf(d2);

        if ((node->maxX - node->minX) / dist < THETA)
        {
            float f = G_CONST * node->mass / (d2 * dist);
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
        else
        {
            for (int k = 0; k < 8; k++)
            {
                int child = node->children[k];
                if (child >= n)
                    stack[ptr++] = child;
                else if (child != -1)
                {
                    float ldx = posX[child] - px, ldy = posY[child] - py, ldz = posZ[child] - pz;
                    float ld2 = ldx * ldx + ldy * ldy + ldz * ldz + SOFTENING;
                    float ldist = sqrtf(ld2);
                    float lf = G_CONST * 10.0f / (ld2 * ldist);
                    ax += lf * ldx;
                    ay += lf * ldy;
                    az += lf * ldz;
                }
            }
        }
    }
    accX[i] = ax;
    accY[i] = ay;
    accZ[i] = az;
}

__global__ void integrateKernel(float *posX, float *posY, float *posZ, float *velX, float *velY, float *velZ, const float *accX, const float *accY, const float *accZ, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    velX[i] = (velX[i] + accX[i] * DT) * DAMPING;
    velY[i] = (velY[i] + accY[i] * DT) * DAMPING;
    velZ[i] = (velZ[i] + accZ[i] * DT) * DAMPING;
    posX[i] += velX[i] * DT;
    posY[i] += velY[i] * DT;
    posZ[i] += velZ[i] * DT;
}

int main()
{
    int N = 50000;
    int iterations = 1000;

    // Pronalaženje prvog većeg stepena broja 2 za Bitonic sort
    int N_padded = 1;
    while (N_padded < N)
        N_padded <<= 1;

    printf("Starting N-Body Benchmark (Barnes-Hut + Bitonic Sort)\n");
    printf("N = %d bodies (Padded to %d for sort), %d iterations\n\n", N, N_padded, iterations);

    float *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ, *d_accX, *d_accY, *d_accZ, *d_mass, *d_bounds;
    unsigned int *d_mortonCodes;
    int *d_indices, *d_nodeCounter, *d_leafNodeIdx;
    OctreeNode *d_nodes;

    cudaMalloc(&d_posX, N * 4);
    cudaMalloc(&d_posY, N * 4);
    cudaMalloc(&d_posZ, N * 4);
    cudaMalloc(&d_velX, N * 4);
    cudaMalloc(&d_velY, N * 4);
    cudaMalloc(&d_velZ, N * 4);
    cudaMalloc(&d_accX, N * 4);
    cudaMalloc(&d_accY, N * 4);
    cudaMalloc(&d_accZ, N * 4);
    cudaMalloc(&d_mass, N * 4);
    cudaMalloc(&d_nodes, (N * 2) * sizeof(OctreeNode));
    cudaMalloc(&d_nodeCounter, 4);
    cudaMalloc(&d_leafNodeIdx, N * 4);
    cudaMalloc(&d_bounds, 24);
    cudaMalloc(&d_mortonCodes, N_padded * 4);
    cudaMalloc(&d_indices, N_padded * 4);

    std::vector<float> h_pos(N);
    for (int i = 0; i < N; i++)
        h_pos[i] = ((float)rand() / RAND_MAX - 0.5f) * 1000.0f;
    cudaMemcpy(d_posX, h_pos.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_pos.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, h_pos.data(), N * 4, cudaMemcpyHostToDevice);

    std::vector<float> h_m(N, 10.0f);
    cudaMemcpy(d_mass, h_m.data(), N * 4, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0;
    for (int i = 0; i < iterations; i++)
    {
        cudaEventRecord(start);

        int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        computeBoundingBoxKernel<<<1, 1>>>(d_posX, d_posY, d_posZ, N, d_bounds);

        // Inicijalizacija ključeva na max (padding) i indeksa
        cudaMemset(d_mortonCodes, 0xFF, N_padded * 4);
        computeMortonCodesKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ, d_bounds, d_mortonCodes, d_indices, N);

        // --- BITONIC SORT ---
        int sortBlocks = N_padded / THREADS_PER_BLOCK;
        for (int k = 2; k <= N_padded; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                bitonicSortKernel<<<sortBlocks, THREADS_PER_BLOCK>>>(d_mortonCodes, d_indices, j, k);
            }
        }

        cudaMemset(d_nodes, 0, (N * 2) * sizeof(OctreeNode));
        buildOctreeKernel<<<1, 1>>>(d_posX, d_posY, d_posZ, d_nodes, N, d_nodeCounter, d_bounds, d_leafNodeIdx, d_indices);

        int hCount;
        cudaMemcpy(&hCount, d_nodeCounter, 4, cudaMemcpyDeviceToHost);

        computeCOMKernel<<<blocks, THREADS_PER_BLOCK>>>(d_nodes, d_posX, d_posY, d_posZ, d_mass, N, d_leafNodeIdx);
        finalizeCOMKernel<<<(hCount + 255) / 256, 256>>>(d_nodes, hCount);
        computeForceKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ, d_accX, d_accY, d_accZ, d_nodes, N);
        integrateKernel<<<blocks, THREADS_PER_BLOCK>>>(d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, d_accX, d_accY, d_accZ, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;

        if (i % 10 == 0)
            printf("Iteration %d: %.2f ms\n", i, ms);
    }

    printf("\nBenchmark Results:\n");
    printf("Average Time: %.3f ms\n", totalTime / iterations);
    printf("Estimated Max FPS: %.1f\n", 1000.0f / (totalTime / iterations));

    return 0;
}