// ============================================================================
// Lab 5 - TDDD56: Multicore and GPU Programming
// Part 1: Naive Box Filter (baseline)
// ============================================================================
//
// This is the baseline implementation of a 2D box filter (low-pass convolution)
// on the GPU using CUDA. Each thread computes one output pixel by averaging
// all pixels in a (2*radius+1) x (2*radius+1) neighborhood.
//
// The load balance was fixed (from 1 thread/block to 16x16 threads/block)
// to properly utilize the GPU's parallelism. This version serves as the
// performance reference for comparing the optimized versions.
//
// Key characteristics:
//   - No shared memory usage (all reads go to global memory)
//   - Simple and readable kernel
//   - Suboptimal due to redundant global memory reads
//
// Compile:
//   cd src/naive && nvcc filter.cu -o ../../filter ../common/milli.cpp \
//       ../common/readppm.cpp -lGL -lglut
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


__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	if (x < imagesizex && y < imagesizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
		{
			// Use max and min to avoid branching!
			int yy = min(max(y+dy, 0), imagesizey-1);
			int xx = min(max(x+dx, 0), imagesizex-1);
			
			sumx += image[((yy)*imagesizex+(xx))*3+0];
			sumy += image[((yy)*imagesizex+(xx))*3+1];
			sumz += image[((yy)*imagesizex+(xx))*3+2];
		}
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

	/*
	// Mesurer le temps avec mauvais load balance
    printf("\n=== Test with bad load balance (1 thread per block) ===\n");
    ResetMilli();
    dim3 grid(imagesizex, imagesizey);
    filter<<<grid, 1>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); 
    cudaDeviceSynchronize();
    double time1 = GetMicroseconds() / 1000.0; // Convert to milliseconds
    printf("Execution time (bad): %.3f ms\n", time1);*/
    
    // Measure time with good load balance
    printf("\n=== Test with good load balance (16x16 threads per block) ===\n");
    ResetMilli();
    dim3 blockSize(16, 16);
    dim3 gridSize((imagesizex + blockSize.x - 1) / blockSize.x, 
                  (imagesizey + blockSize.y - 1) / blockSize.y);
    filter<<<gridSize, blockSize>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey);
    cudaDeviceSynchronize();
    double time2 = GetMicroseconds() / 1000.0; // Convert to milliseconds
    printf("Execution time (good): %.3f ms\n", time2);
    
    // Afficher le speedup
    //printf("\nSpeedup: %.2fx speeder\n", time1 / time2);


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
	printf("╔══════════════════════════════════╗\n");
	printf("║  CUDA Image Filter - Naive Test  ║\n");
	printf("╚══════════════════════════════════╝\n");
	
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
