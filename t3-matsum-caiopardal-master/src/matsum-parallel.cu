#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <omp.h>

__global__ void matrix_sum(int *C, int *A, int *B, int rows, int cols, int dim) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  int index = row * cols + col;
  if((row < rows && col < cols) && (index < dim)) {
      C[index] = A[index] + B[index];
  }
}

int main(int argc, char **argv) {
  int *A, *B, *C;
  int i, j;
  double t;

  int *d_a, *d_b, *d_c;

  // Input
  int rows, cols;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return EXIT_FAILURE;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return EXIT_FAILURE;
  }

  fscanf(input, "%d", &rows);
  fscanf(input, "%d", &cols);

  int size = sizeof(int) * rows * cols;

  // Allocate memory on the host
  A = (int *)malloc(size);
  B = (int *)malloc(size);
  C = (int *)malloc(size);

  // Alloc device arrays
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Initialize memory
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      A[i * cols + j] = B[i * cols + j] = i + j;
    }
  }


  // Copy data to device
  cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

  // Compute matrix sum on device
  // Leave only the kernel and synchronize inside the timing region!
  t = omp_get_wtime();
  
  // Init dimGrid and dimBlocks
  dim3 dimGrid(ceil((float)cols / 16), ceil((float)rows / 16), 1);
  dim3 dimBlock(16, 16, 1);

  // Call function
  matrix_sum<<<dimGrid, dimBlock>>>(d_c, d_a, d_b, rows, cols, rows * cols);
  cudaDeviceSynchronize();
  t = omp_get_wtime() - t;

  // Copy data back to host
  cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

  long long int sum = 0;

  // Keep this computation on the CPU
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      sum += C[i * cols + j];
    }
  }

  fprintf(stdout, "%lli\n", sum);
  fprintf(stderr, "%lf\n", t);

  free(A);
  free(B);
  free(C);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
