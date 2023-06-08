#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "../include/helper_cuda.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

#define DBG_CHECK { printf("DBG_CHECK: file %s at line %d\n", __FILE__, __LINE__ ); }
#define DEBUG  // without debug (with random imputs) the kernel does not work

#define NPROBS 3
#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype float

#define RUN_SOLUTIONS

#define PRINT_MATRIX(A, N, M, ST ) {  \
      int i, j;  \
      printf("%s:\n", ( ST ));  \
      for (i=0; i< ( N ); i++) {  \
        printf("\t");  \
        for (j=0; j< ( M ); j++)  \
          printf("%6.3f ", A[i*( M ) + j]);  \
        printf("\n");  \
      }  \
      printf("\n\n");  \
}

float matrix_error (int n, int m, const dtype* A, const dtype* B) {
  int i, j;
  dtype error = (dtype)0;
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      error += fabs(B[i*m + j] - A[i*m + j]);

  return(error);
}

#define CEIL_DIV( N, D ) ((( N ) % ( D )) == 0) ? (( N )/( D )) : ((( N )/( D ))+1)

#define BLK_EDGE 32     // sgemm_naive

// ----------------------------------------


int verbose;

__global__ void sgemm_naive(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < N && y < M) {
    dtype tmp = 0.0;
    for (int i = 0; i < K; ++i)
      tmp += A[x * K + i] * B[i * M + y];

    // C = α*(A@B)+β*C
    C[x * M + y] = alpha * tmp + beta * C[x * M + y];
  }
}

// ================================= Write here your Kernel =================================


__global__ void my_GEMM_kernel(int N, int K, int M, float alpha, const dtype *A, const dtype *B, float beta, dtype *C) {

        /*  [ ... your GEMM kernel ... ]  */

}


// ==========================================================================================

dtype* execute_gemm_kernel (int n, int k, int m, float alpha, dtype* A, dtype* B, float beta, void (*gemm_kernel)(int, int, int, float, const dtype*, const dtype*, float, dtype*), float* Bandwidth, float* CompTime, double* Flops) {
  int grd_sizeX, grd_sizeY;
  int blk_sizeX, blk_sizeY;

  // ---------------------------------
  if (gemm_kernel == sgemm_naive) {
    grd_sizeX = CEIL_DIV(n, BLK_EDGE);
    grd_sizeY = CEIL_DIV(m, BLK_EDGE);

    blk_sizeX = BLK_EDGE;
    blk_sizeY = BLK_EDGE;
  } else {
    /*
     ---------- WRITE HERE YOUR KERNEL PARAMETERS ----------

     grd_sizeX = ??? ;
     grd_sizeY = ??? ;

     blk_sizeX = ??? ;
     blk_sizeY = ??? ;

     -------------------------------------------------------
     */
  }
  // ---------------------------------

  // ------------------- allocating GPU vectors ----------------------
  dtype *dev_A, *dev_B, *dev_C;

  checkCudaErrors( cudaMalloc(&dev_A, n*k*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_B, k*m*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_C, n*m*sizeof(dtype)) );
  size_t bandwidth_numerator = ((n*k) + (k*m) + (n*m))*sizeof(dtype);

  // ----------------- copy date from host to device -----------------

  checkCudaErrors( cudaMemcpy(dev_A, A, n*k*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dev_B, B, k*m*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemset(dev_C, 0, n*m*sizeof(dtype)) );

  // ---------- compute GPU_tmp_b with the reduction kernel ----------
  TIMER_DEF;
  TIMER_START;

  {
      dim3 block_size(blk_sizeX, blk_sizeY, 1);
      dim3 grid_size(grd_sizeX, grd_sizeY, 1);
      printf("%d: block_size = (%d, %d), grid_size = (%d, %d)\n", __LINE__, block_size.x, block_size.y, grid_size.x, grid_size.y);
      gemm_kernel<<<grid_size, block_size>>>(n, k, m, alpha, (const dtype*)dev_A, (const dtype*)dev_B, beta, dev_C);
  }


  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  *CompTime += TIMER_ELAPSED;
  *Bandwidth = bandwidth_numerator / ((*CompTime)*1e+9);
  *Flops  = (n) / ((*CompTime)*1e+9);
  *Flops *= m;
  *Flops *= k;
  *Flops *= 2;

  // --------------- copy results from device to host ----------------

  dtype *GPU_C = (dtype*)malloc(sizeof(dtype)*n*m);
  checkCudaErrors( cudaMemcpy(GPU_C, dev_C, n*m*sizeof(dtype), cudaMemcpyDeviceToHost) );

  if (verbose > 1)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C form execute_gemm_kernel")

  checkCudaErrors( cudaFree(dev_A) );
  checkCudaErrors( cudaFree(dev_B) );
  checkCudaErrors( cudaFree(dev_C) );

  return(GPU_C);
}


int main(int argc, char *argv[]) {

  printf("====================================== Problem computations ======================================\n");
  // =========================================== Set-up the problem ============================================

  if (argc < 3) {
    printf("Usage: lab3_ex1 e v [CPU_ON = 1]\n");
    return(1);
  }
  printf("argv[1] = %s\n", argv[1]);
  printf("argv[2] = %s\n", argv[2]);
  if (argc > 3)
    printf("argv[3] = %s\n", argv[3]);

  // ---------------- set-up the problem size -------------------


  int e = atoi(argv[1]), n = (1<<(e/2)), k = n, m = n, i, j, CPU_ON = 1;
  float alpha = 1.0f, beta = 1.0f;
  verbose = atoi(argv[2]);
  if (argc > 3)
    CPU_ON = atoi(argv[3]);

  // BUG check: code to generalize
  if ((e%2) != 0 || e<=10) {
    printf("Now the code only support squared matrices. So, since the generated matrix will have dimensions 2^(e/2) x 2^(e/2), e must be even\n");
    printf("Furthermore, due to lenght reasons, e must be >=12\n");
    exit(42);
  }

  printf("e = %d --> n = k = m = 2^(e/2) = %d\n", e, n);
  printf("alpha = %f, beta = %f\n", alpha, beta);
  printf("CPU_ON = %d\n", CPU_ON);
  printf("verbose = %d\n", verbose);
  printf("dtype = %s\n", XSTR(dtype));

  // ======================================== Get the device properties ========================================
  printf("======================================= Device properties ========================================\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:                              %d bit\n",
           deviceProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:                     %7.3f GB/s\n",
           2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);

    printf("  Peak Arithmetic Intensity:                     %7.3f GFLOPS/s\n",
           2.0*deviceProp.memoryClockRate*(_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount)/1.0e6);

  }

  // ------------------ set-up the timers ---------------------

  TIMER_DEF;
  const char* lables[NPROBS] = {"CPU check", "Example Kernel", "Your Kernel" };
  float errors[NPROBS], Times[NPROBS], Bandwidths[NPROBS], error;
  double Flops[NPROBS];
  for (i=0; i<NPROBS; i++) {
    errors[i] = 1<<30;
    Bandwidths[i] = 0;
    Flops[i] = 0;
    Times[i] = 0;
  }


  // ------------------- set-up the problem -------------------

  dtype *A, *B, *GPU_C, *CPU_C;
  A = (dtype*)malloc(sizeof(dtype)*n*k);
  B = (dtype*)malloc(sizeof(dtype)*k*m);
  CPU_C = (dtype*)malloc(sizeof(dtype)*n*m);
//   GPU_C = (dtype*)malloc(sizeof(dtype)*n*m);

  time_t t;
  srand((unsigned) time(&t));


  for (i=0; i<(n*k); i++)
    A[i] = ((dtype)(i/m)/(dtype)m) + 1.0f;
  for (i=0; i<(k*m); i++)
    B[i] = (dtype)(1);

#ifdef DEBUG
  if (verbose > 0) {
    PRINT_MATRIX(A, n, k, "A")

    PRINT_MATRIX(B, k, m, "B")
  }
#endif
  // ======================================== Running the computations =========================================

  /* [ ... ]
   */

  // ========================== CPU computation =========================
  if (CPU_ON) {

    TIMER_START;
    for (i=0; i<n; i++)
      for (j=0; j<m; j++)
        for (int h=0; h<k; h++)
          CPU_C[i*m +j] += A[i*k + h] * B[h*m + j];
    TIMER_STOP;

    Times[0] = TIMER_ELAPSED;
    errors[0] = 0.0f;
    Bandwidths[0] = 0.0f;
    Flops[0]  = (n) / (Times[0]*1e+9);
    Flops[0] *= m;
    Flops[0] *= k;


    if (verbose > 0)
      PRINT_MATRIX(CPU_C, n, m, "CPU_C")

  } else {
    Times[0] = -1.0f;
    errors[0] = -1.0f;
    Bandwidths[0] = -1.0f;
    Flops[0] = -1.0f;
  }
  printf("=========================== Example GPU Kernel ===========================\n");
  // =========================== Example GPU Kernel ===========================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, sgemm_naive, &Bandwidths[1], &Times[1], &Flops[1]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[1] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);

  printf("============================ Your GPU Kernel =============================\n");
  // ============================ Your GPU Kernel =============================

  GPU_C = execute_gemm_kernel(n, k, m, alpha, A, B, beta, my_GEMM_kernel, &Bandwidths[2], &Times[2], &Flops[2]);

  // ------------- Compare GPU and CPU solution --------------

  (CPU_ON) ? (error = matrix_error(n, m, CPU_C, GPU_C)) : (error = 0.0f) ;
  errors[2] = error;

  if (verbose > 0)
    PRINT_MATRIX(GPU_C, n, m, "GPU_C")

  free(GPU_C);


  printf("\n\n");
  if (!(CPU_ON)) printf("CPU check not lunched!!\n");
  printf("Solution\n %9s\t%9s\t%9s\t%16s\t%16s\n", "type", "error", "time (s)", "flops (GFLOPS/s)", "bandwidth (GB/s)");
  for (int i=0; i<NPROBS; i++) {
    if ((i != 6))
      printf("%12s:\t%9.6f\t%9.6f\t%16.6lf\t%16.6f\n", lables[i], errors[i], Times[i], Flops[i], Bandwidths[i]);
  }
  printf("\n");

  printf("GPU times: e Kernel1_time Kernel1_flops Kernel2_time Kernel2_flops ... on stderr\n");
  fprintf(stderr, "%d, ", e);
  for (i=1; i<NPROBS; i++)
    fprintf(stderr, "%f, %f, ", Times[i], Flops[i]);
  fprintf(stderr, "\n");

  return(0);
}
