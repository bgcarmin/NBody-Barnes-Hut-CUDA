# N-Body Simulation with CUDA and Barnes-Hut Algorithm

A high-performance N-body gravitational simulation using NVIDIA CUDA with Barnes-Hut octree optimization for efficient force calculation.

## Overview

This project simulates gravitational interactions between particles using the Barnes-Hut algorithm, which reduces force computation from O(n²) to O(n log n) by organizing particles in a spatial octree structure.

## Versions

### nbody_v5.cu
Interactive visualization version with real-time OpenGL rendering.
- **Particles:** 500,000
- **Features:** 
  - Barnes-Hut octree acceleration structure
  - Real-time 3D visualization with GLEW/FreeGLUT
  - Mouse camera control (rotation and zoom)
  - Morton code spatial sorting
  - Dynamic particle physics with velocity and acceleration

### nbody_v5_bench.cu
Benchmarking version for performance analysis without visualization.
- **Particles:** 1,000,000
- **Features:**
  - Same Barnes-Hut algorithm as v5
  - Detailed timing metrics
  - CPU-GPU memory transfer benchmarks
  - Kernel execution profiling
  - Performance analysis output

## Compilation

### With Visualization (nbody_v5.cu)
```bash
nvcc -o nbody_v5 nbody_v5.cu -arch=sm_75 -O3 -std=c++17 -lfreeglut -lglew32 -lopengl32 -lglu32
```

### Benchmark Version (nbody_v5_bench.cu)
```bash
nvcc -o nbody_v5_bench nbody_v5_bench.cu -arch=sm_75 -O3 -std=c++17
```

## Running

### Interactive Simulation
```bash
./nbody_v5.exe
```
- **Mouse:** Click and drag to rotate camera
- **Scroll:** Zoom in/out

### Benchmark
```bash
./nbody_v5_bench.exe
```
Outputs performance metrics for:
- Octree construction
- Force calculations
- Particle position updates
- Memory transfers

## Key Parameters

- **THETA:** 0.5f (Barnes-Hut angle threshold for multipole approximation)
- **G_CONST:** 0.5f (Gravitational constant, reduced for stability)
- **DT:** 0.02f (Time step)
- **SOFTENING:** 50.0f (Softening parameter to prevent singularities)
- **MAX_SPEED:** 500.0f (Speed cap for stability)

## Algorithm Details

### Barnes-Hut Octree
1. **Compute Morton Codes:** Spatial sorting of particles using Morton Z-order curves
2. **Build Octree:** Parallel octree construction using particle-batch insertion
3. **Compute Centers of Mass:** Calculate mass and center of mass for each octree node
4. **Force Calculation:** For each particle, traverse octree applying multipole approximation

### Force Computation
Uses gravitational force with softening to prevent singularities:
$$F = \frac{G \cdot m_1 \cdot m_2}{(r^2 + \text{SOFTENING})^{3/2}} \cdot \vec{r}$$

## Hardware Requirements

- NVIDIA GPU with compute capability ≥ 7.5 (Tesla V100, RTX series, etc.)
- CUDA Toolkit 11.0+
- For visualization: OpenGL 3.3+, GLEW, FreeGLUT

## Todo/Improvements

- [ ] Parallelize octree construction
- [ ] Parallelize bounding box computation with parallel reduction
- [ ] Make max tree depth dynamic instead of fixed
