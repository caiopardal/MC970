#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// Function that sorts a vector `a` of size `n`
double count_sort(double a[], int n, int nt) {
  int i, j, count;
  double *temp;
  double start, end, duracao;

  temp = (double *)malloc(n * sizeof(double));

  start = omp_get_wtime();
#pragma omp parallel for num_threads(nt) \ 
    default(none) private(count, i, j) shared(a, n, temp)
  for (i = 0; i < n; i++) {
    count = 0;
    for (j = 0; j < n; j++)
      if ((a[j] < a[i]) || ((a[j] == a[i]) && (j < i)))
        count++;
    temp[count] = a[i];
  }
  end = omp_get_wtime();

  duracao = end - start;

  memcpy(a, temp, n * sizeof(double));
  free(temp);

  return duracao;
}

int main(int argc, char *argv[]) {
  int i, n, nt;
  double *a, t_s;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return 1;
  }

  fscanf(input, "%d", &nt);

  // Read the number of elements
  fscanf(input, "%d", &n);

  // Do not change this line
  omp_set_num_threads(nt);

  // Allocate the vector to be ordered
  a = (double *)malloc(n * sizeof(double));

  // Populate the vector
  for (i = 0; i < n; i++)
    fscanf(input, "%lf", &a[i]);

  // Execute Counting Sort
  t_s = count_sort(a, n, nt);

  // Print the ordered vector
  for (i = 0; i < n; i++)
    printf("%.2lf ", a[i]);

  printf("\n");

  // Print the time it took to sort
  fprintf(stderr, "%lf\n", t_s);

  return 0;
}
