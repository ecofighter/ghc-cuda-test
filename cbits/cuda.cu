#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"

#define CHECK(call) \
{ \
  const cudaError_t error = call; \
  if(error != cudaSuccess) \
  { \
    printf("Error: %s: %d, ", __FILE__, __LINE__); \
    printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
  } \
}

extern "C" __global__ void dotp(double* A, double* B, double* out){
  int bx = blockIdx.x;
  int bdx = blockDim.x;
  int gdx = gridDim.x;
  int tx =  threadIdx.x;
  int BLOCK_SIZE = 32;

  double a[20], b[20];

  /*** Global A,B -> Register a,b ***/
#pragma unroll
  for(int i = 0; i < 20; i++){
    a[i] = A[20*BLOCK_SIZE*bx + tx + BLOCK_SIZE*i];
  }

#pragma unroll
  for(int i = 0; i < 20; i++){
    b[i] = B[20*BLOCK_SIZE*bx + tx + BLOCK_SIZE*i];
  }

  /**** dot production ***/
  double o = 0;
  for(int i = 0; i < 20; ++i){
    o += a[i]*b[i];
  }

  /**** Register o -> Global out ***/
  out[bx*BLOCK_SIZE + tx] = o;
}

extern "C" void dot(int size){
  int N = size;
  int T = 32;
//  int T = atoi(argv[2]);

  double * A, *B, *out;
  A = (double*)malloc( N*N*N*20*sizeof(double));
  B = (double*)malloc( N*N*N*20*sizeof(double) );
  out = (double*)malloc( N*N*N*sizeof(double));


  // initialize
  for(int i = 0; i < N*N*N; ++i){
    for(int j = 0; j < 20; ++j){
      A[i*20 + j] = 100*i+j;
      B[i*20 + j] = 1000*i+j;
    }
    out[i] = 0.0;
  }


  double *dA, *dB, *dout;
  cudaMalloc( (void**)&dA, N*N*N*20*sizeof(double));
  cudaMalloc( (void**)&dB, N*N*N*20*sizeof(double));
  cudaMalloc( (void**)&dout, N*N*N*sizeof(double));

  cudaMemcpy(A, dA, N*N*N*20*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(B, dB, N*N*N*20*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(out, dout, N*N*N*sizeof(double), cudaMemcpyHostToDevice);

  dim3 grid(N*N*N/T);
  dim3 block(T);

//  StartTimer();
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
  dotp<<<grid,block>>>(dA,dB,dout);
  CHECK(cudaDeviceSynchronize());

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
float time;
cudaEventElapsedTime(&time, start, stop);

//  double time = GetTimer(); // [ms]
  double flops = 39*N*N*N / (time * 1e-3); // Flop/sec
  printf("%d^3: time %f[ms], flops %f [GFlops]\n", N, time, flops * 1e-9);

  cudaMemcpy(A, dA, N*N*N*20*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(B, dB, N*N*N*20*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(out, dout, N*N*N*sizeof(double), cudaMemcpyDeviceToHost);

  free(out);
  free(A);
  free(B);

  cudaFree(dout);
  cudaFree(dA);
  cudaFree(dB);
}
