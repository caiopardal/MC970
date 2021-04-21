#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define HISTOGRAM_SIZE 64
#define TILE_WITDH 16

void check_cuda(cudaError_t error, const char *filename, const int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

__global__ void histogram_kernel(PPMImage *d_image, float* hist) {
  __shared__ float private_hist[HISTOGRAM_SIZE];
    float size = d_image->y*d_image->x*1.0;

    if(threadIdx.x * TILE_WITDH + threadIdx.y < HISTOGRAM_SIZE) private_hist[threadIdx.x * TILE_WITDH + threadIdx.y] = 0;
    __syncthreads();

    // Get variable values
    int col = blockDim.x * blockIdx.x + threadIdx.x,
        row = blockDim.y * blockIdx.y + threadIdx.y,
        index = row * d_image->x + col;

    if((row < d_image->y && col < d_image->x) && (index < d_image->x*d_image->y)) {
      atomicAdd(&(private_hist[16*d_image->data[index].red + 4 * d_image->data[index].green + d_image->data[index].blue]), 1);
    }

    __syncthreads();
    if(threadIdx.x * TILE_WITDH + threadIdx.y < HISTOGRAM_SIZE) {
      atomicAdd(&(hist[threadIdx.x * TILE_WITDH + threadIdx.y]), (private_hist[threadIdx.x * TILE_WITDH + threadIdx.y]/size));
    }
}

double Histogram(PPMImage *image, float *h_h) {
  float ms;
  cudaEvent_t start, stop;
  
  int i;
  unsigned int rows, cols, img_size;
  PPMImage *d_image;
  PPMPixel *d_pixels;
  float *d_hist;

  // Create Events
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  // Get image data
  cols = image->x;
  rows = image->y;
  img_size = cols * rows;

  //Process every image data
  for (i = 0; i < img_size; i++) {
    image->data[i].red = floor((image->data[i].red * 4) / 256);
    image->data[i].blue = floor((image->data[i].blue * 4) / 256);
    image->data[i].green = floor((image->data[i].green * 4) / 256);
  }

  cudaMalloc((void **)&d_image, sizeof(PPMImage));
  cudaMalloc((void **)&d_pixels, sizeof(PPMPixel) * img_size);
  cudaMalloc((void **)&d_hist, HISTOGRAM_SIZE*sizeof(float));
  
  cudaMemcpy(d_image, image, sizeof(PPMImage), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pixels, image->data, sizeof(PPMPixel) * img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_image->data), &d_pixels, sizeof(PPMPixel *), cudaMemcpyHostToDevice);

  cudaMemcpy(d_hist, h_h, HISTOGRAM_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil((float)cols / TILE_WITDH), ceil((float)rows / TILE_WITDH), 1);
  dim3 dimBlock(TILE_WITDH, TILE_WITDH, 1);
  
  // Launch kernel and compute kernel runtime.
  // Warning: make sure only the kernel is being profiled, memcpies should be
  // out of this region.
  CUDACHECK(cudaEventRecord(start));
  histogram_kernel<<<dimGrid, dimBlock>>>(d_image, d_hist);
  CUDACHECK(cudaEventRecord(stop));
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&ms, start, stop));

  cudaMemcpy(h_h, d_hist, HISTOGRAM_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // Destroy events
  CUDACHECK(cudaEventDestroy(start));
  CUDACHECK(cudaEventDestroy(stop));

  cudaFree(d_image);
  cudaFree(d_pixels);
  cudaFree(d_hist);

  return ((double)ms) / 1000.0;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  PPMImage *image = readPPM(argv[1]);
  float *h = (float *)malloc(sizeof(float) * 64);

  // Initialize histogram
  for (int i = 0; i < 64; i++)
    h[i] = 0.0;

  // Compute histogram
  double t = Histogram(image, h);

  for (int i = 0; i < 64; i++)
    printf("%0.3f ", h[i]);
  printf("\n");

  fprintf(stderr, "%lf\n", t);
  free(h);
}
