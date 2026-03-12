# Image Filtering with CUDA

## Table of Contents

- [Part 1: Load Balance Fix](#part-1-load-balance-fix)
- [Part 2: Box Filter with Shared Memory](#part-2-box-filter-with-shared-memory)
- [Part 3: Separable Low-Pass Filter](#part-3-separable-low-pass-filter)
- [Part 4: Gaussian Filter with Kernel Weights](#part-4-gaussian-filter-with-kernel-weights)
- [Part 5: Median Filter](#part-5-median-filter)

---

## Part 1: Load Balance Fix

The original code launched the kernel with `dim3 grid(imagesizex, imagesizey)` and only **1 thread per block**, wasting the GPU's massive parallelism. The fix uses 16×16 thread blocks with a properly computed grid:

```cuda
dim3 blockSize(16, 16);
dim3 gridSize((imagesizex + blockSize.x - 1) / blockSize.x,
              (imagesizey + blockSize.y - 1) / blockSize.y);
```

This ensures all pixels are covered while utilizing 256 threads per block.

---

## Part 2: Box Filter with Shared Memory

### How much data is in shared memory?

For a block of 16×16 threads with a kernel of radius `r`, the shared memory size is:

```
(BLOCK_SIZE_X + 2*r) × (BLOCK_SIZE_Y + 2*r) × 3 bytes
```

| Kernel | Radius | Shared memory tile | Size per block |
|--------|--------|--------------------|----------------|
| 7×7    | 3      | 22 × 22 × 3       | 1,452 bytes    |
| 21×21  | 10     | 36 × 36 × 3       | 3,888 bytes    |

The shared memory includes both the output region (16×16) and the halo regions needed for the convolution.

### Strategy for copying to shared memory

A **distributed strided loading** strategy is used:

1. Each thread gets a unique ID: `threadId = threadIdx.y × blockDim.x + threadIdx.x` (0 to 255)
2. Total pixels to load: `totalSharedPixels = sharedWidth × sharedHeight`
3. Threads iterate with stride: `for(int i = threadId; i < totalSharedPixels; i += threadsPerBlock)`
4. At each iteration, the thread converts the linear index to 2D shared coordinates
5. Maps to global coordinates and clamps to image boundaries

This ensures balanced work distribution and coalesced memory access.

### How much data does each thread copy?

For a 16×16 block (256 threads) with a 7×7 kernel (radius = 3):

- Total pixels to load: 22 × 22 = 484 pixels
- Pixels per thread: 484 / 256 ≈ **1.89 pixels** (most load 2, some load 1)

For a 21×21 kernel: 36 × 36 = 1,296 pixels → **~5 pixels per thread**.

The work is automatically balanced through the strided loop pattern.

### Handling overlap between blocks

The halo region is handled by:

1. **Extending the loading region**: each block loads a region larger than its output area, shifted by the kernel radius:
   ```
   blockStartX = blockIdx.x × blockDim.x - kernelsizex
   blockStartY = blockIdx.y × blockDim.y - kernelsizey
   ```
2. **Clamping at image boundaries**: pixels outside the image are clamped to the nearest valid coordinate
3. **Offset in shared memory**: output pixels are accessed at `(threadIdx + kernelsize)` in shared memory

### Maximum safe block size

Maximum safe block size is approximately **24×24 = 576 threads**.

Constraints:
- **Thread limit**: GPUs allow a maximum of 1024 threads per block (32×32 = 1024)
- **Shared memory**: for a 24×24 block with max kernel (radius=10): (24+20)×(24+20)×3 = 5,808 bytes — well within the ~48 KB limit
- **Register pressure**: larger blocks mean fewer registers per thread

24×24 provides a good balance between parallelism and resource usage.

### Performance results

Results on Olympen (512×512 image):

| Kernel | Naive     | Shared Memory | Speedup |
|--------|-----------|---------------|---------|
| 9×9    | 0.187 ms  | 0.152 ms      | 1.23×   |
| 21×21  | 1.021 ms  | 0.747 ms      | 1.37×   |

The speedup is modest because modern GPUs (Tesla/Ampere) have efficient L1/L2 caches that already handle repeated global memory reads well. Shared memory would show larger gains on older GPUs, very large images, or much larger kernels.

### Memory access coalescing

**Yes**, global memory access is coalesced:

- **Loading phase**: threads with consecutive IDs load pixels at consecutive memory addresses (row-major layout: `image[(y*width + x)*3]`)
- **Output phase**: adjacent threads write to consecutive addresses (`out[(y*width + x)*3]`)

The strided loop maintains coalescing because threads still access consecutive locations in each iteration.

---

## Part 3: Separable Low-Pass Filter

### Performance: separable vs. non-separated

| Kernel | Non-separable (shared) | Separable (H + V) | Speedup |
|--------|------------------------|--------------------|---------|
| 9×9    | 0.191 ms               | 0.114 ms           | 1.68×   |
| 21×21  | 0.945 ms               | 0.165 ms           | 5.73×   |

The separable filter reduces the number of operations from $O(N^2)$ to $O(2N)$ per pixel. For a 21×21 kernel, this means 441 operations reduced to 42 — a theoretical **10.5× reduction**. The measured 5.73× speedup confirms a significant real-world benefit.

---

## Part 4: Gaussian Filter with Kernel Weights

### Visual comparison: box vs. Gaussian

**Yes**, the Gaussian-filtered image is noticeably better:

- **Box filter**: uniform blur with visible blockiness and harsh transitions
- **Gaussian filter**: smooth, natural-looking blur with softer transitions that better preserves image structure

### Timing comparison

| Filter type                                     | Execution time |
|-------------------------------------------------|----------------|
| Non-separable box (shared memory), 9×9          | 0.191 ms       |
| Separable box, 9×9                              | 0.114 ms       |
| Separable Gaussian `[1,2,4,8,16,8,4,2,1]`, 9×9 | 0.117 ms       |

The Gaussian filter is only marginally slower than the box filter despite performing weighted multiplications — the extra computation is negligible compared to memory access time.

### Delivering custom weights to the GPU

Three approaches, from best to worst for this use case:

| Method | Pros | Cons |
|--------|------|------|
| **`__constant__` memory** + `cudaMemcpyToSymbol` | Fast cached broadcast access; read-only | Fixed size; not dynamic during execution |
| **Global memory parameter** (cudaMalloc + pointer) | Flexible size; easy to change | Slower access; needs explicit management |
| **Shared memory copy per block** | Fast after initial load; reduces global traffic | Adds loading overhead; limited size |

For small, fixed-size kernels, **constant memory** is the best choice.

---

## Part 5: Median Filter

### Algorithm for finding the median

A **bubble sort** is used on each thread's local array of neighborhood values:

```cuda
for (i = 0; i < count - 1; i++)
    for (j = 0; j < count - i - 1; j++)
        if (values[j] > values[j+1])
            swap(values[j], values[j+1]);
return values[count / 2];
```

Bubble sort is simple and efficient enough for very small arrays (up to 21 elements for our maximum kernel size). More sophisticated algorithms (e.g., partial sort, sorting networks) would be beneficial for larger kernels.

### Best filter size for noise reduction

**3×3 to 5×5** provides the best balance between noise removal and detail preservation. Larger kernels (7×7+) introduce too much blurring and loss of detail, while 3×3 may not fully remove heavy noise.

The separable median is not mathematically equivalent to a full 2D median, but provides a good approximation at a fraction of the cost.
