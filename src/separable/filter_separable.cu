// ============================================================================
// Lab 5 - TDDD56: Multicore and GPU Programming
// Part 3: Separable Box Filter
// ============================================================================
//
// Separable filter optimization: instead of applying a full NxN 2D kernel,
// we split it into two 1D passes (horizontal 1xN, then vertical Nx1).
// This reduces the number of operations from O(N^2) to O(2N) per pixel.
//
// Each pass uses shared memory with a halo region in only one direction,
// requiring less shared memory than the full 2D approach.
//
// The mathematical equivalence holds because the box filter is separable:
//   conv2D(image, box_NxN) = conv1D_V(conv1D_H(image, box_1xN), box_Nx1)
//
// Compile:
//   cd src/separable && nvcc filter_separable.cu -o ../../filter_separable \
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

// Horizontal filter (1×N) - processes rows
__global__ void filterHorizontal(unsigned char *image, unsigned char *out, 
                                  const unsigned int imagesizex, const unsigned int imagesizey, 
                                  const int kernelsizex)
{
	// Shared memory: only need horizontal halo
	__shared__ unsigned char sharedMem[BLOCK_SIZE_Y][BLOCK_SIZE_X + 2*maxKernelSizeX][3];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int localX = threadIdx.x + kernelsizex;
	int localY = threadIdx.y;
	
	int sharedWidth = BLOCK_SIZE_X + 2*kernelsizex;
	int sharedHeight = BLOCK_SIZE_Y;
	
	int totalSharedPixels = sharedWidth * sharedHeight;
	int threadsPerBlock = blockDim.x * blockDim.y;
	int threadId = threadIdx.y * blockDim.x + threadIdx.x;
	
	int blockStartX = blockIdx.x * blockDim.x - kernelsizex;
	int blockStartY = blockIdx.y * blockDim.y;
	
	// Load data into shared memory (horizontal strip with halo)
	for(int i = threadId; i < totalSharedPixels; i += threadsPerBlock)
	{
		int sharedY = i / sharedWidth;
		int sharedX = i % sharedWidth;
		
		int globalX = blockStartX + sharedX;
		int globalY = blockStartY + sharedY;
		
		// Clamp to image boundaries
		globalX = min(max(globalX, 0), (int)imagesizex - 1);
		globalY = min(max(globalY, 0), (int)imagesizey - 1);
		
		if (globalY < imagesizey && globalX < imagesizex)
		{
			sharedMem[sharedY][sharedX][0] = image[(globalY * imagesizex + globalX) * 3 + 0];
			sharedMem[sharedY][sharedX][1] = image[(globalY * imagesizex + globalX) * 3 + 1];
			sharedMem[sharedY][sharedX][2] = image[(globalY * imagesizex + globalX) * 3 + 2];
		}
	}
	
	__syncthreads();
	
	// Apply horizontal filter
	if (x < imagesizex && y < imagesizey)
	{
		unsigned int sumx = 0, sumy = 0, sumz = 0;
		int divby = 2*kernelsizex + 1;
		
		// Convolve along X axis only
		for(int dx = -kernelsizex; dx <= kernelsizex; dx++)
		{
			int sharedX = localX + dx;
			
			sumx += sharedMem[localY][sharedX][0];
			sumy += sharedMem[localY][sharedX][1];
			sumz += sharedMem[localY][sharedX][2];
		}
		
		out[(y*imagesizex+x)*3+0] = sumx/divby;
		out[(y*imagesizex+x)*3+1] = sumy/divby;
		out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

// Vertical filter (N×1) - processes columns
__global__ void filterVertical(unsigned char *image, unsigned char *out, 
                                const unsigned int imagesizex, const unsigned int imagesizey, 
                                const int kernelsizey)
{
	// Shared memory: only need vertical halo
	__shared__ unsigned char sharedMem[BLOCK_SIZE_Y + 2*maxKernelSizeY][BLOCK_SIZE_X][3];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int localX = threadIdx.x;
	int localY = threadIdx.y + kernelsizey;
	
	int sharedWidth = BLOCK_SIZE_X;
	int sharedHeight = BLOCK_SIZE_Y + 2*kernelsizey;
	
	int totalSharedPixels = sharedWidth * sharedHeight;
	int threadsPerBlock = blockDim.x * blockDim.y;
	int threadId = threadIdx.y * blockDim.x + threadIdx.x;
	
	int blockStartX = blockIdx.x * blockDim.x;
	int blockStartY = blockIdx.y * blockDim.y - kernelsizey;
	
	// Load data into shared memory (vertical strip with halo)
	for(int i = threadId; i < totalSharedPixels; i += threadsPerBlock)
	{
		int sharedY = i / sharedWidth;
		int sharedX = i % sharedWidth;
		
		int globalX = blockStartX + sharedX;
		int globalY = blockStartY + sharedY;
		
		// Clamp to image boundaries
		globalX = min(max(globalX, 0), (int)imagesizex - 1);
		globalY = min(max(globalY, 0), (int)imagesizey - 1);
		
		if (globalY < imagesizey && globalX < imagesizex)
		{
			sharedMem[sharedY][sharedX][0] = image[(globalY * imagesizex + globalX) * 3 + 0];
			sharedMem[sharedY][sharedX][1] = image[(globalY * imagesizex + globalX) * 3 + 1];
			sharedMem[sharedY][sharedX][2] = image[(globalY * imagesizex + globalX) * 3 + 2];
		}
	}
	
	__syncthreads();
	
	// Apply vertical filter
	if (x < imagesizex && y < imagesizey)
	{
		unsigned int sumx = 0, sumy = 0, sumz = 0;
		int divby = 2*kernelsizey + 1;
		
		// Convolve along Y axis only
		for(int dy = -kernelsizey; dy <= kernelsizey; dy++)
		{
			int sharedY = localY + dy;
			
			sumx += sharedMem[sharedY][localX][0];
			sumy += sharedMem[sharedY][localX][1];
			sumz += sharedMem[sharedY][localX][2];
		}
		
		out[(y*imagesizex+x)*3+0] = sumx/divby;
		out[(y*imagesizex+x)*3+1] = sumy/divby;
		out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

// Global variables for image data
unsigned char *image, *pixels, *dev_bitmap, *dev_input, *dev_temp;
unsigned int imagesizey, imagesizex; // Image size
bool buffersAllocated = false; // Track if GPU buffers are allocated

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsize)
{
	if (kernelsize > maxKernelSizeX || kernelsize > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	// Allocate buffers only once
	if (!buffersAllocated)
	{
		pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
		cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
		cudaMalloc( (void**)&dev_temp, imagesizex*imagesizey*3);
		cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
		buffersAllocated = true;
	}
	
	// Always copy fresh input data and clear buffers
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMemset( dev_temp, 0, imagesizex*imagesizey*3 ); // Clear temp buffer
	cudaMemset( dev_bitmap, 0, imagesizex*imagesizey*3 ); // Clear output buffer

	dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 gridSize((imagesizex + blockSize.x - 1) / blockSize.x, 
	              (imagesizey + blockSize.y - 1) / blockSize.y);

	int filterSize = 2*kernelsize + 1;
	printf("\n=== Separable Filter: %dx%d = (%dx1) * (1x%d) ===\n", 
	       filterSize, filterSize, filterSize, filterSize);
	
	// Start timing
	ResetMilli();
	
	// STEP 1: Apply horizontal filter (1×N)
	filterHorizontal<<<gridSize, blockSize>>>(dev_input, dev_temp, imagesizex, imagesizey, kernelsize);
	cudaDeviceSynchronize();
	
	// STEP 2: Apply vertical filter (N×1) on the result
	filterVertical<<<gridSize, blockSize>>>(dev_temp, dev_bitmap, imagesizex, imagesizey, kernelsize);
	cudaDeviceSynchronize();
	
	// End timing
	double timeSeparable = GetMicroseconds() / 1000.0;
	printf("Execution time: %.3f ms\n", timeSeparable);
	
	// Calculate theoretical operations
	int opsFullFilter = imagesizex * imagesizey * filterSize * filterSize;
	int opsSeparable = imagesizex * imagesizey * 2 * filterSize;
	printf("Operations: Full=%d, Separable=%d, Reduction=%.2fx\n", 
	       opsFullFilter, opsSeparable, (float)opsFullFilter/opsSeparable);

	// Check for errors!
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
		
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	
	// Don't free here - buffers are reused across multiple calls
}

// Display images
void Draw()
{
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	}
	else
	{
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
	glutCreateWindow("Lab 5 - Separable Filter");
	glutDisplayFunc(Draw);

	ResetMilli();

	printf("\n\n");
	printf("╔═══════════════════════════════════════════╗\n");
	printf("║  CUDA Filter - SEPARABLE FILTER Version  ║\n");
	printf("╚═══════════════════════════════════════════╝\n");
	
	printf("\n>>> Testing with 3x3 kernel (radius=1) <<<");
	computeImages(1);
	
	printf("\n>>> Testing with 5x5 kernel (radius=2) <<<");
	computeImages(2);
	
	printf("\n>>> Testing with 7x7 kernel (radius=3) <<<");
	computeImages(3);
	
	printf("\n>>> Testing with 9x9 kernel (radius=4) <<<");
	computeImages(4);
	
	printf("\n>>> Testing with 11x11 kernel (radius=5) <<<");
	computeImages(5);
	
	printf("\n>>> Testing with 13x13 kernel (radius=6) <<<");
	computeImages(6);
	
	printf("\n>>> Testing with 15x15 kernel (radius=7) <<<");
	computeImages(7);
	
	printf("\n>>> Testing with 17x17 kernel (radius=8) <<<");
	computeImages(8);
	
	printf("\n>>> Testing with 19x19 kernel (radius=9) <<<");
	computeImages(9);
	
	printf("\n>>> Testing with 21x21 kernel (radius=10 - MAX) <<<");
	computeImages(10);

	// Clean up GPU memory before starting OpenGL
	if (buffersAllocated)
	{
		cudaFree( dev_bitmap );
		cudaFree( dev_temp );
		cudaFree( dev_input );
		buffersAllocated = false;
	}

	glutMainLoop();
	
	// Final cleanup
	if (pixels) free(pixels);
	
	return 0;
}
