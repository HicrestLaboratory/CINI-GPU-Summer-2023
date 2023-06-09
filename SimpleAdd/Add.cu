#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Create the event for getting the time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate Unified Memory accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Record the event
  cudaEventRecord(start);
  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);
  // Wait for GPU to finish before accessing on host
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  // Altarntivelly you can you cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Measure the time recorded
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel Time (ms): " << milliseconds << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
