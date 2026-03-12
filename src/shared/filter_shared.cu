// ============================================================================
// Lab 5 - TDDD56: Multicore and GPU Programming
// Part 2: Box Filter with Shared Memory Optimization
// ============================================================================
//
// Optimized version of the naive box filter that preloads image data into
// shared memory before performing the convolution. Each block loads a tile
// of pixels (including the halo/overlap region) into fast on-chip shared
// memory, reducing redundant global memory accesses.
//
// Strategy:
//   1. Each block is responsible for a BLOCK_SIZE x BLOCK_SIZE output patch
//   2. Shared memory includes the output region + halo (kernel radius)
//   3. All threads cooperatively load data using a strided loop pattern
//   4. After __syncthreads(), each thread computes its output from shared mem
//
// Shared memory per block: (BLOCK_SIZE + 2*radius)^2 * 3 bytes (RGB)
//   - 7x7 kernel:  22x22x3 = 1,452 bytes
//   - 21x21 kernel: 36x36x3 = 3,888 bytes
//
// Compile:
//   cd src/shared && nvcc filter_shared.cu -o ../../filter_shared \
//       ../common/milli.cpp ../common/readppm.cpp -lGL -lglut
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "../common/readppm.h"
#include "../common/milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Optimized filter with shared memory
__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
	// Shared memory: block size + radius (halo) on all sides
	// Each pixel has 3 channels (RGB)
	__shared__ unsigned char sharedMem[BLOCK_SIZE_Y + 2*maxKernelSizeY][BLOCK_SIZE_X + 2*maxKernelSizeX][3];
	
	// Global position of this thread's output pixel
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Local position in shared memory (offset by radius for halo)
	int localX = threadIdx.x + kernelsizex;
	int localY = threadIdx.y + kernelsizey;
	
	// Dimensions of shared memory region
	int sharedWidth = BLOCK_SIZE_X + 2*kernelsizex;
	int sharedHeight = BLOCK_SIZE_Y + 2*kernelsizey;
	
	// Total pixels to load into shared memory
	int totalSharedPixels = sharedWidth * sharedHeight;
	int threadsPerBlock = blockDim.x * blockDim.y;
	
	// Each thread gets a unique ID within the block
	int threadId = threadIdx.y * blockDim.x + threadIdx.x;
	
	// Starting position for this block in global memory (top-left corner including halo)
	int blockStartX = blockIdx.x * blockDim.x - kernelsizex;
	int blockStartY = blockIdx.y * blockDim.y - kernelsizey;
	
	// LOADING PHASE: Each thread loads multiple pixels to fill shared memory
	// This ensures balanced work distribution and coalesced memory access
	for(int i = threadId; i < totalSharedPixels; i += threadsPerBlock)
	{
		// Convert linear index to 2D coordinates in shared memory
		int sharedY = i / sharedWidth;
		int sharedX = i % sharedWidth;
		
		// Calculate corresponding global coordinates
		int globalX = blockStartX + sharedX;
		int globalY = blockStartY + sharedY;
		
		// Clamp to image boundaries (handle edges)
		globalX = min(max(globalX, 0), (int)imagesizex - 1);
		globalY = min(max(globalY, 0), (int)imagesizey - 1);
		
		// Load pixel data (coalesced access - threads with consecutive IDs access consecutive memory)
		sharedMem[sharedY][sharedX][0] = image[(globalY * imagesizex + globalX) * 3 + 0];
		sharedMem[sharedY][sharedX][1] = image[(globalY * imagesizex + globalX) * 3 + 1];
		sharedMem[sharedY][sharedX][2] = image[(globalY * imagesizex + globalX) * 3 + 2];
	}
	
	// Synchronize to ensure all threads have finished loading data
	__syncthreads();
	
	// COMPUTATION PHASE: Apply filter using shared memory
	if (x < imagesizex && y < imagesizey)
	{
		unsigned int sumx = 0, sumy = 0, sumz = 0;
		int divby = (2*kernelsizex+1) * (2*kernelsizey+1);
		
		// Apply box filter kernel - now reading from fast shared memory!
		for(int dy = -kernelsizey; dy <= kernelsizey; dy++)
		{
			for(int dx = -kernelsizex; dx <= kernelsizex; dx++)
			{
				int sharedY = localY + dy;
				int sharedX = localX + dx;
				
				sumx += sharedMem[sharedY][sharedX][0];
				sumy += sharedMem[sharedY][sharedX][1];
				sumz += sharedMem[sharedY][sharedX][2];
			}
		}
		
		// Write output (coalesced write)
		out[(y*imagesizex+x)*3+0] = sumx/divby;
		out[(y*imagesizex+x)*3+1] = sumy/divby;
		out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size
bool buffersAllocated = false; // Track if GPU buffers are allocated

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	// Allocate buffers only once
	if (!buffersAllocated)
	{
		pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
		cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
		cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
		buffersAllocated = true;
	}
	
	// Always copy fresh input data and clear output buffer
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMemset( dev_bitmap, 0, imagesizex*imagesizey*3 ); // Clear output buffer!

	dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 gridSize((imagesizex + blockSize.x - 1) / blockSize.x, 
	              (imagesizey + blockSize.y - 1) / blockSize.y);

	// Calculate shared memory usage
	int sharedMemSize = (BLOCK_SIZE_X + 2*kernelsizex) * (BLOCK_SIZE_Y + 2*kernelsizey) * 3;
	
	// printf("\n========================================\n");
	// printf("Image: %dx%d pixels\n", imagesizex, imagesizey);
	// printf("Filter: %dx%d (radius=%d)\n", 2*kernelsizex+1, 2*kernelsizey+1, kernelsizex);
	// printf("Block size: %dx%d threads (%d threads/block)\n", 
	//        BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_X*BLOCK_SIZE_Y);
	// printf("Grid size: %dx%d blocks\n", gridSize.x, gridSize.y);
	// printf("Shared memory per block: %d bytes\n", sharedMemSize);
	// printf("Total shared memory: %.2f KB\n", (sharedMemSize * gridSize.x * gridSize.y) / 1024.0);
	// printf("========================================\n");
    
    // Test optimized version with shared memory
    printf("\n=== Optimized filter (WITH shared memory) ===\n");
    ResetMilli();
    filter<<<gridSize, blockSize>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
    cudaDeviceSynchronize();
    double timeShared = GetMicroseconds() / 1000.0;
    printf("Execution time: %.3f ms\n", timeShared);

//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	
	// Don't free here - buffers are reused across multiple calls
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"images/input/maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	// Test with different kernel sizes to see the speedup difference
	printf("\n\n");
	printf("╔════════════════════════════════════════════════════════╗\n");
	printf("║  CUDA Image Filter - Shared Memory Optimization Test  ║\n");
	printf("╚════════════════════════════════════════════════════════╝\n");
	
	printf("\n>>> Testing with 3x3 kernel (radius=1) <<<\n");
	computeImages(1, 1);
	
	printf("\n>>> Testing with 5x5 kernel (radius=2) <<<\n");
	computeImages(2, 2);
	
	printf("\n>>> Testing with 7x7 kernel (radius=3) <<<\n");
	computeImages(3, 3);
	
	printf("\n>>> Testing with 9x9 kernel (radius=4) <<<\n");
	computeImages(4, 4);
	
	printf("\n>>> Testing with 11x11 kernel (radius=5) <<<\n");
	computeImages(5, 5);
	
	printf("\n>>> Testing with 13x13 kernel (radius=6) <<<\n");
	computeImages(6, 6);
	
	printf("\n>>> Testing with 15x15 kernel (radius=7) <<<\n");
	computeImages(7, 7);
	
	printf("\n>>> Testing with 17x17 kernel (radius=8) <<<\n");
	computeImages(8, 8);
	
	printf("\n>>> Testing with 19x19 kernel (radius=9) <<<\n");
	computeImages(9, 9);
	
	printf("\n>>> Testing with 21x21 kernel (radius=10 - MAX) <<<\n");
	computeImages(10, 10);

	// Clean up GPU memory before starting OpenGL
	if (buffersAllocated)
	{
		cudaFree( dev_bitmap );
		cudaFree( dev_input );
		buffersAllocated = false;
	}

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	
	// Final cleanup
	if (pixels) free(pixels);
	
	return 0;
}
