// nbody_cuda_simple.cu - O(N^2) verzija bez octree (za testiranje)
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
#define G_CONST 1.0f
#define DT 0.03f
#define SOFTENING 0.001f

// ============================================================================
// KERNEL: O(N^2) FORCE CALCULATION
// ============================================================================

__global__ void computeForceN2Kernel(
    const float *posX, const float *posY, const float *posZ,
    const float *mass,
    float *accX, float *accY, float *accZ,
    int numBodies
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= numBodies) return;
    
    float px = posX[i];
    float py = posY[i];
    float pz = posZ[i];
    
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    
    // Compute force from all other bodies
    for (int j = 0; j < numBodies; j++) {
        if (i == j) continue;
        
        float dx = posX[j] - px;
        float dy = posY[j] - py;
        float dz = posZ[j] - pz;
        
        float distSq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING;
        float invDist = rsqrtf(distSq);  // 1/sqrt(distSq)
        float invDist3 = invDist * invDist * invDist;
        
        float f = G_CONST * mass[j] * invDist3;
        
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }
    
    accX[i] = ax;
    accY[i] = ay;
    accZ[i] = az;
}

// ============================================================================
// KERNEL: INTEGRATION (Velocity Verlet)
// ============================================================================

__global__ void integrateKernel(
    float *posX, float *posY, float *posZ,
    float *velX, float *velY, float *velZ,
    const float *accX, const float *accY, const float *accZ,
    int numBodies,
    float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= numBodies) return;
    
    // Update velocity
    velX[i] += accX[i] * dt;
    velY[i] += accY[i] * dt;
    velZ[i] += accZ[i] * dt;
    
    // Update position
    posX[i] += velX[i] * dt;
    posY[i] += velY[i] * dt;
    posZ[i] += velZ[i] * dt;
}

// ============================================================================
// ERROR CHECKING
// ============================================================================

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        
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
    // Initialize CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "CUDA Error: No CUDA-capable device detected!\n");
        fprintf(stderr, "Error code: %d - %s\n", error, cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    
    cudaSetDevice(0);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("========================================\n");
    printf("CUDA N-Body Simulation (O(N^2) Method)\n");
    printf("========================================\n");
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("========================================\n");
    
    const int N = 500000;
    const int NUM_STEPS = 10;
    
    printf("Number of bodies: %d\n", N);
    printf("Number of time steps: %d\n", NUM_STEPS);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Time step (dt): %.3f\n", DT);
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
        
        // Circular velocity
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
    
    // Calculate grid dimensions
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Grid dimensions: %d blocks x %d threads\n", numBlocks, THREADS_PER_BLOCK);
    
    printf("\nStarting simulation...\n\n");
    
    // Main simulation loop
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < NUM_STEPS; step++) {
        auto stepStart = std::chrono::high_resolution_clock::now();
        
        // Compute forces (O(N^2))
        computeForceN2Kernel<<<numBlocks, THREADS_PER_BLOCK>>>(
            d_posX, d_posY, d_posZ, d_mass,
            d_accX, d_accY, d_accZ, N
        );
        checkCudaError(cudaDeviceSynchronize(), "Force calculation");
        
        // Integrate positions and velocities
        integrateKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
            d_posX, d_posY, d_posZ,
            d_velX, d_velY, d_velZ,
            d_accX, d_accY, d_accZ,
            N, DT
        );
        checkCudaError(cudaDeviceSynchronize(), "Integration");
        
        auto stepEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> stepTime = stepEnd - stepStart;
        
        if (step % 10 == 0 || step < 5) {
            printf("Step %4d/%d | Time: %8.2f ms\n", step, NUM_STEPS, stepTime.count());
        }
    }
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = totalEnd - totalStart;
    
    printf("\n========================================\n");
    printf("Simulation completed!\n");
    printf("Total time: %.2f seconds\n", totalTime.count());
    printf("Average time per step: %.2f ms\n", totalTime.count() * 1000.0 / NUM_STEPS);
    printf("Performance: %.2f GFLOP/s\n", (20.0 * N * N * NUM_STEPS) / (totalTime.count() * 1e9));
    printf("========================================\n");
    
    // Copy final results back
    printf("\nCopying results back to host...\n");
    checkCudaError(cudaMemcpy(h_posX, d_posX, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posX");
    checkCudaError(cudaMemcpy(h_posY, d_posY, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posY");
    checkCudaError(cudaMemcpy(h_posZ, d_posZ, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy posZ");
    checkCudaError(cudaMemcpy(h_velX, d_velX, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy velX");
    checkCudaError(cudaMemcpy(h_velY, d_velY, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy velY");
    checkCudaError(cudaMemcpy(h_velZ, d_velZ, N * sizeof(float), cudaMemcpyDeviceToHost), "final copy velZ");
    
    // Calculate and print some statistics
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
    printf("\nSaving results to output.txt...\n");
    FILE *fp = fopen("output.txt", "w");
    if (fp) {
        fprintf(fp, "# Final positions and velocities after %d steps\n", NUM_STEPS);
        fprintf(fp, "# Format: x y z vx vy vz\n");
        for (int i = 0; i < N; i++) {
            fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f\n", 
                    h_posX[i], h_posY[i], h_posZ[i],
                    h_velX[i], h_velY[i], h_velZ[i]);
        }
        fclose(fp);
        printf("Results saved successfully!\n");
    } else {
        fprintf(stderr, "Warning: Could not open output.txt for writing\n");
    }
    
    // Cleanup
    printf("\nCleaning up...\n");
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
