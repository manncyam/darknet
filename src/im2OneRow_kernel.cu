#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "im2OneRow.h"

// convert rgb to 1D vector with all values of r, g and b in order
__global__ void im2OneRow_kernel(unsigned char* dev_src, float* dev_dst, int channels, int rows, int cols, int step)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int k = blockIdx.y;

	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
	int rgb_index = i * step + j * channels + k;

	dev_dst[index] = dev_src[rgb_index] / 255.0f;
}

// used to launch kernel code
void im2OneRow(unsigned char* dev_src, float* dev_dst, int channels, int rows, int cols, int step)
{
	dim3 dg(rows, channels, 1);
	dim3 db(cols, 1, 1);
	// Launch a kernel on the GPU with one thread for each element.
	im2OneRow_kernel << < dg, db >> >(dev_src, dev_dst, channels, rows, cols, step);
}