#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>

struct Matrix {
    int num_rows, num_cols, nnz;
    int *h_rowptr, *h_colind, *d_rowptr, *d_colind;
    double *h_vals, *d_vals;
};

void MatrixVectorMultGPU(struct Matrix *A, int A_m, int A_n, 
                         double *d_x, int x_m, double *d_y);

void VectorAddGPU(double *u, double *v, double a, double *w, int n);
double VectorDotGPU(double *u, double *v, int n);
void ConjugateGradient(struct Matrix *A, int A_m, int A_n, double *b, double *x, int max_iter, double eps);

#endif
