#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MASK_WIDTH 15
#define CUDA_GRID 16

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

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

  cudaMallocHost(&img, sizeof(PPMImage));
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

void writePPM(PPMImage *img) {

  fprintf(stdout, "P6\n");
  fprintf(stdout, "# %s\n", COMMENT);
  fprintf(stdout, "%d %d\n", img->x, img->y);
  fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

  fwrite(img->data, 3 * img->x, img->y, stdout);
  fclose(stdout);
}

__global__ void smoothing_kernel(PPMPixel *image, PPMPixel *image_copy, int img_x, int img_y) {
  int x, y;
  int p_x = threadIdx.x + blockDim.x * blockIdx.x;
  int p_y = threadIdx.y + blockDim.y * blockIdx.y;

  // Error prevention
  if (p_x >= img_x || p_y >= img_y)
    return;

  __shared__ PPMPixel img_shrd[CUDA_GRID+MASK_WIDTH-1][CUDA_GRID+MASK_WIDTH-1];

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int local_x = p_x - (MASK_WIDTH-1)/2; local_x <= p_x; local_x++) {
      for (int local_y = p_y-(MASK_WIDTH-1)/2; local_y <= p_y; local_y++) {
        int shared_x = local_x + (MASK_WIDTH-1)/2 - p_x;
        int shared_y = local_y + (MASK_WIDTH-1)/2 - p_y;
        int global_unique_value = (local_y * img_x) + local_x;

        if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
          img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
          img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
          img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
        } else {
          img_shrd[shared_x][shared_y].red = 0;
          img_shrd[shared_x][shared_y].green = 0;
          img_shrd[shared_x][shared_y].blue = 0;
        }
      } 
    }
  } else if (threadIdx.x == CUDA_GRID - 1 && threadIdx.y == 0) {
    for (int local_x = p_x; local_x <= p_x + (MASK_WIDTH - 1)/2; local_x++) {
      for (int local_y = p_y-(MASK_WIDTH-1)/2; local_y <= p_y; local_y++) {
        int shared_x = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_x - p_x);
        int shared_y = local_y + (MASK_WIDTH-1)/2 - p_y;
        int global_unique_value = (local_y * img_x) + local_x;

        if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
          img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
          img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
          img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
        } else {
          img_shrd[shared_x][shared_y].red = 0;
          img_shrd[shared_x][shared_y].green = 0;
          img_shrd[shared_x][shared_y].blue = 0;
        }
      }
    }
  } else if (threadIdx.x == 0 && threadIdx.y == CUDA_GRID - 1) {
    for (int local_x = p_x - (MASK_WIDTH-1)/2; local_x <= p_x; local_x++) {
      for (int local_y = p_y; local_y <= p_y + (MASK_WIDTH - 1)/2; local_y++) {
        int shared_x = local_x + (MASK_WIDTH-1)/2 - p_x;
        int shared_y = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_y - p_y);
        int global_unique_value = (local_y * img_x) + local_x;

        if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
          img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
          img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
          img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
        } else {
          img_shrd[shared_x][shared_y].red = 0;
          img_shrd[shared_x][shared_y].green = 0;
          img_shrd[shared_x][shared_y].blue = 0;
        }
      }
    }
  } else if (threadIdx.x == CUDA_GRID - 1 && threadIdx.y == CUDA_GRID - 1) {
    for (int local_x = p_x; local_x <= p_x + (MASK_WIDTH - 1)/2; local_x++) {
      for (int local_y = p_y; local_y <= p_y + (MASK_WIDTH - 1)/2; local_y++) {
        int shared_x = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_x - p_x);
        int shared_y = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_y - p_y);
        int global_unique_value = (local_y * img_x) + local_x;

        if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
          img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
          img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
          img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
        } else {
          img_shrd[shared_x][shared_y].red = 0;
          img_shrd[shared_x][shared_y].green = 0;
          img_shrd[shared_x][shared_y].blue = 0;
        }
      }
    }
  } else if (threadIdx.x == 0) {
    for (int local_x = p_x - (MASK_WIDTH-1)/2; local_x <= p_x; local_x++) {
      int shared_x = local_x + (MASK_WIDTH-1)/2 - p_x;
      int shared_y = (MASK_WIDTH-1)/2  + threadIdx.y;
      int global_unique_value = (p_y * img_x) + local_x;

      if (local_x >= 0 && local_x < img_x) {
        img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
        img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
        img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
      } else {
          img_shrd[shared_x][shared_y].red = 0;
          img_shrd[shared_x][shared_y].green = 0;
          img_shrd[shared_x][shared_y].blue = 0;
      }
    }
  } else if (threadIdx.x == CUDA_GRID - 1) {
    for (int local_x = p_x; local_x <= p_x + (MASK_WIDTH - 1)/2; local_x++) {
      int shared_x = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_x - p_x);
      int shared_y = (MASK_WIDTH-1)/2  + threadIdx.y;
      int global_unique_value = (p_y * img_x) + local_x;

      if (local_x >= 0 && local_x < img_x) {
        img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
        img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
        img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
      } else {
        img_shrd[shared_x][shared_y].red = 0;
        img_shrd[shared_x][shared_y].green = 0;
        img_shrd[shared_x][shared_y].blue = 0;
      }
    }
  } else if (threadIdx.y == 0) {
    for (int local_y = p_y-(MASK_WIDTH-1)/2; local_y <= p_y; local_y++) {
      int shared_x = (MASK_WIDTH-1)/2 + threadIdx.x;
      int shared_y = local_y + (MASK_WIDTH-1)/2 - p_y;
      int global_unique_value = (local_y * img_x) + p_x;

      if (local_y >= 0 && local_y < img_y) {
        img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
        img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
        img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
      } else {
        img_shrd[shared_x][shared_y].red = 0;
        img_shrd[shared_x][shared_y].green = 0;
        img_shrd[shared_x][shared_y].blue = 0;
      }
    }
  } else if (threadIdx.y == CUDA_GRID - 1) {
    for (int local_y = p_y; local_y <= p_y + (MASK_WIDTH - 1)/2; local_y++) {
      int shared_x = (MASK_WIDTH-1)/2 + threadIdx.x;
      int shared_y = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_y - p_y);
      int global_unique_value = (local_y * img_x) + p_x;

      if (local_y >= 0 && local_y < img_y) {
        img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
        img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
        img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
      } else {
        img_shrd[shared_x][shared_y].red = 0;
        img_shrd[shared_x][shared_y].green = 0;
        img_shrd[shared_x][shared_y].blue = 0;
      }
    }
  } else {
    int shared_x = threadIdx.x+(MASK_WIDTH-1)/2;
    int shread_y = threadIdx.y+ (MASK_WIDTH-1)/2;

    img_shrd[shared_x][shread_y].red = image_copy[(p_y * img_x) + p_x].red;
    img_shrd[shared_x][shread_y].green = image_copy[(p_y * img_x) + p_x].green;
    img_shrd[shared_x][shread_y].blue = image_copy[(p_y * img_x) + p_x].blue;
  }
  
  __syncthreads();

  int total_red = 0 , total_blue = 0, total_green = 0;

  for (y = threadIdx.y; y <= threadIdx.y + MASK_WIDTH-1; y++) {
    for (x = threadIdx.x; x <= threadIdx.x + MASK_WIDTH-1; x++) {
      if (x >= 0 && y >= 0 && y < CUDA_GRID+MASK_WIDTH && x < CUDA_GRID+MASK_WIDTH) {
        total_red += img_shrd[x][y].red;
        total_blue += img_shrd[x][y].blue;
        total_green += img_shrd[x][y].green;
      }
    }
  }

  image[(p_y * img_x) + p_x].red = total_red / (MASK_WIDTH*MASK_WIDTH);
  image[(p_y * img_x) + p_x].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
  image[(p_y * img_x) + p_x].green = total_green / (MASK_WIDTH*MASK_WIDTH);
}

void Smoothing(PPMImage *image, PPMImage *image_copy) {
  int n = (image_copy->x * image_copy->y);
  PPMPixel *d_image ,*d_image_output, *img_out;

  // Allocate the memory
  img_out = (PPMPixel *)calloc(n, sizeof(PPMPixel));
  cudaMalloc((void **) &d_image, sizeof(PPMPixel) * n);
  cudaMalloc((void **) &d_image_output, sizeof(PPMPixel) * n);

  // Copy the image data
  cudaMemcpy(d_image, image_copy->data, sizeof(PPMPixel)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_image_output, image_copy->data, sizeof(PPMPixel)*n, cudaMemcpyHostToDevice);    

  int nrow = ceil((double) image_copy->x / CUDA_GRID);    
  int nlin = ceil((double) image_copy->y / CUDA_GRID);    
  dim3 gridDim(nrow, nlin); // Iterate on a grid of 16x16
  dim3 blockDim(CUDA_GRID, CUDA_GRID);

  smoothing_kernel<<<gridDim, blockDim>>>(d_image_output, d_image, image_copy->x, image_copy->y);
  cudaDeviceSynchronize();

  cudaMemcpy(img_out, d_image_output, sizeof(PPMPixel) * n, cudaMemcpyDeviceToHost); 
  image->data = img_out;
}

int main(int argc, char *argv[]) {
  FILE *input;
  char filename[255];
  double t;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  // Read input filename
  fscanf(input, "%s\n", filename);

  // Read input file
  PPMImage *image = readPPM(filename);
  PPMImage *image_output = readPPM(filename);

  // Call Smoothing Kernel
  t = omp_get_wtime();
  Smoothing(image_output, image);
  t = omp_get_wtime() - t;

  // Write result to stdout
  writePPM(image_output);

  // Print time to stderr
  fprintf(stderr, "%lf\n", t);

  // Cleanup
  cudaFreeHost(image);
  cudaFreeHost(image_output);

  return 0;
}
