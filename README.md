# CUDA Image Filtering

**Course**: TDDD56 — Multicore and GPU Programming (Linköping University)

## Overview

This lab explores GPU-accelerated image filtering using CUDA. Starting from a naive convolution filter, we progressively optimize it through shared memory utilization, separable filter decomposition, and weighted (Gaussian) kernels. A median filter is also implemented for noise reduction.

All filters read PPM images, process them on the GPU, and display the original / filtered result side by side using OpenGL/GLUT.

## Project Structure

```
lab5/
├── README.md                  # This file
├── report.md                  # Answers to all lab questions with benchmarks
├── Makefile                   # Build all filters at once
├── kernel_on_image.pdf        # how the blocs of the image are computed with the kernel
│
├── src/
│   ├── common/                # Shared utilities
│   │   ├── readppm.cpp/.h     #   PPM image reader/writer
│   │   └── milli.cpp/.h       #   Microsecond-precision timer
│   │
│   ├── naive/                 # Part 1 — Baseline box filter
│   │   └── filter.cu          #   Naive 2D convolution (no shared memory)
│   │
│   ├── shared/                # Part 2 — Shared memory optimization
│   │   └── filter_shared.cu   #   Tiled box filter with cooperative loading
│   │
│   ├── separable/             # Part 3 — Separable (H+V) box filter
│   │   └── filter_separable.cu
│   │
│   ├── gaussian/              # Part 4 — Gaussian weighted filters
│   │   ├── filter_gaussian.cu                #   Separable Gaussian (1×9 / 9×1)
│   │   └── filter_gaussian_non_separable.cu  #   Full 2D Gaussian (reference)
│   │
│   └── median/                # Part 5 — Median filter (noise reduction)
│       └── filter_median.cu   #   Separable median with bubble sort
│
└── images/
    ├── input/                 # Test images (PPM format)
    │   ├── maskros512.ppm     #   512×512 dandelion photo
    │   ├── baboon1.ppm        #   512×512 baboon (classic reference)
    │   ├── img1.ppm           #   1280×640 photo
    │   ├── maskros-noisy.ppm  #   Noisy version of maskros512
    │   └── img1-noisy.ppm     #   Noisy version of img1
    └── output/                # Generated output images
        ├── out.ppm
        └── out_gaussian.ppm
```

## Building

### Prerequisites

- NVIDIA CUDA Toolkit (nvcc compiler)
- OpenGL and GLUT development libraries (`libgl-dev`, `freeglut3-dev`)

### Using the Makefile

```bash
make              # Build all filter variants
make naive        # Build only the naive filter
make shared       # Build only the shared memory filter
make separable    # Build only the separable filter
make gaussian     # Build only the Gaussian separable filter
make gaussian_ns  # Build only the non-separable Gaussian filter
make median       # Build only the median filter
make clean        # Remove all compiled binaries
```

### Manual compilation (example)

```bash
cd src/naive
nvcc filter.cu -o ../../filter ../common/milli.cpp ../common/readppm.cpp -lGL -lglut
```

## Running

Each binary accepts an optional PPM image path as an argument:

```bash
./filter                              # Uses default image (maskros512.ppm)
./filter images/input/baboon1.ppm     # Specify a different image
./filter_median images/input/maskros-noisy.ppm  # Denoise an image
```

The program opens a window showing the original and filtered image side by side, and prints timing results to the terminal.

## Lab Parts Summary

| Part | File | Description | Key Concept |
|------|------|-------------|-------------|
| 1 | `filter.cu` | Fix load balance, naive box filter | Thread block sizing |
| 2 | `filter_shared.cu` | Box filter with shared memory | Tiled loading, halo regions |
| 3 | `filter_separable.cu` | Separable box filter (H + V) | Filter decomposition, O(N²) → O(2N) |
| 4a | `filter_gaussian.cu` | Separable Gaussian filter | Weighted kernels, `__constant__` memory |
| 4b | `filter_gaussian_non_separable.cu` | Full 2D Gaussian (reference) | Outer product of 1D weights |
| 5 | `filter_median.cu` | Separable median filter | Non-linear filter, noise reduction |

## Performance Highlights (512×512 image, Olympen)

| Configuration | 9×9 kernel | 21×21 kernel |
|---------------|------------|--------------|
| Naive (global memory) | 0.187 ms | 1.021 ms |
| Shared memory | 0.152 ms | 0.747 ms |
| Separable box | 0.114 ms | 0.165 ms |
| Separable Gaussian | 0.117 ms | — |

The separable filter provides the largest speedup, reaching **~6× faster** than the non-separated version for large kernels.

## Report

See [report.md](report.md) for detailed answers to all lab questions, including shared memory analysis, performance benchmarks, and implementation strategies.
