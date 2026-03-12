// ============================================================================
// Lab 5 - TDDD56: Multicore and GPU Programming
// Part 5: Separable Median Filter (noise reduction)
// ============================================================================
//
// Non-linear filter that returns the median of pixel values in a neighborhood.
// Unlike convolution filters, the median filter preserves edges while
// effectively removing salt-and-pepper noise.
//
// Implementation:
//   - Separable approach: horizontal 1xN median, then vertical Nx1 median
//   - Not mathematically equivalent to a full 2D median, but a good and
//     much faster approximation (O(2N) vs O(N^2) per pixel)
//   - Uses bubble sort to find median (efficient for small kernel sizes)
//   - Each channel (R, G, B) is processed independently
//
// Best noise reduction results: 3x3 to 5x5 kernel (larger = too much blur)
//
// Compile:
//   cd src/median && nvcc filter_median.cu -o ../../filter_median \
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

// Device function to sort and find median
// Simple bubble sort - works well for small arrays (up to 21 elements)
__device__ unsigned char findMedian(unsigned char* values, int count)
{
	// Bubble sort
	for(int i = 0; i < count - 1; i++)
	{
		for(int j = 0; j < count - i - 1; j++)
		{
			if(values[j] > values[j + 1])
			{
				// Swap
				unsigned char temp = values[j];
				values[j] = values[j + 1];
				values[j + 1] = temp;
			}
		}
	}
	
	// Return median (middle element)
	return values[count / 2];
}

// Horizontal median filter (1×N) - processes rows
__global__ void medianHorizontal(unsigned char *image, unsigned char *out, 
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
	
	// Apply horizontal median filter
	if (x < imagesizex && y < imagesizey)
	{
		int kernelSize = 2*kernelsizex + 1;
		unsigned char valuesR[2*maxKernelSizeX + 1];
		unsigned char valuesG[2*maxKernelSizeX + 1];
		unsigned char valuesB[2*maxKernelSizeX + 1];
		
		// Collect values from neighborhood
		int idx = 0;
		for(int dx = -kernelsizex; dx <= kernelsizex; dx++)
		{
			int sharedX = localX + dx;
			
			valuesR[idx] = sharedMem[localY][sharedX][0];
			valuesG[idx] = sharedMem[localY][sharedX][1];
			valuesB[idx] = sharedMem[localY][sharedX][2];
			idx++;
		}
		
		// Find median for each channel
		out[(y*imagesizex+x)*3+0] = findMedian(valuesR, kernelSize);
		out[(y*imagesizex+x)*3+1] = findMedian(valuesG, kernelSize);
		out[(y*imagesizex+x)*3+2] = findMedian(valuesB, kernelSize);
	}
}

// Vertical median filter (N×1) - processes columns
__global__ void medianVertical(unsigned char *image, unsigned char *out, 
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
	
	// Apply vertical median filter
	if (x < imagesizex && y < imagesizey)
	{
		int kernelSize = 2*kernelsizey + 1;
		unsigned char valuesR[2*maxKernelSizeY + 1];
		unsigned char valuesG[2*maxKernelSizeY + 1];
		unsigned char valuesB[2*maxKernelSizeY + 1];
		
		// Collect values from neighborhood
		int idx = 0;
		for(int dy = -kernelsizey; dy <= kernelsizey; dy++)
		{
			int sharedY = localY + dy;
			
			valuesR[idx] = sharedMem[sharedY][localX][0];
			valuesG[idx] = sharedMem[sharedY][localX][1];
			valuesB[idx] = sharedMem[sharedY][localX][2];
			idx++;
		}
		
		// Find median for each channel
		out[(y*imagesizex+x)*3+0] = findMedian(valuesR, kernelSize);
		out[(y*imagesizex+x)*3+1] = findMedian(valuesG, kernelSize);
		out[(y*imagesizex+x)*3+2] = findMedian(valuesB, kernelSize);
	}
}

// Global variables for image data
unsigned char *image, *pixels, *dev_bitmap, *dev_input, *dev_temp;
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

	int filterSizeX = 2*kernelsizex + 1;
	int filterSizeY = 2*kernelsizey + 1;
	
	printf("\n=== Separable Median Filter: %dx%d = (%dx1) * (1x%d) ===\n", 
	       filterSizeX, filterSizeY, filterSizeX, filterSizeY);
	
	// Start timing
	ResetMilli();
	
	// STEP 1: Apply horizontal median (1×N)
	medianHorizontal<<<gridSize, blockSize>>>(dev_input, dev_temp, imagesizex, imagesizey, kernelsizex);
	cudaDeviceSynchronize();
	
	// STEP 2: Apply vertical median (N×1) on the result
	medianVertical<<<gridSize, blockSize>>>(dev_temp, dev_bitmap, imagesizex, imagesizey, kernelsizey);
	cudaDeviceSynchronize();
	
	// End timing
	double timeMedian = GetMicroseconds() / 1000.0;
	printf("Execution time: %.3f ms\n", timeMedian);

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
		image = readppm((char *)"images/input/maskros-noisy.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5 - Median Filter (Noise Reduction)");
	glutDisplayFunc(Draw);

	ResetMilli();

	printf("\n\n");
	printf("╔═══════════════════════════════════════════╗\n");
	printf("║  CUDA Filter - MEDIAN FILTER Version     ║\n");
	printf("║  Excellent for noise reduction!          ║\n");
	printf("╚═══════════════════════════════════════════╝\n");
	
	printf("\n>>> Testing with 3x3 median kernel <<<");
	computeImages(1, 1);
	
	printf("\n>>> Testing with 5x5 median kernel <<<");
	computeImages(2, 2);
	
	printf("\n>>> Testing with 7x7 median kernel <<<");
	computeImages(3, 3);
	
	printf("\n>>> Testing with 9x9 median kernel <<<");
	computeImages(4, 4);

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
