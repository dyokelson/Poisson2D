#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include <cublas_v2.h>
#include <iostream>

void MatrixMatrixMultGPU(double *A, int A_m, int A_n,
			 double *B, int B_m, int B_n,
			 double *C);

void MatrixVectorMultGPU(double *A, int A_m, int A_n,
			 double *x, int x_m,
			 double *y);

void VectorAddGPU(double *u, double *v, double a, double *w, int n);
double VectorDotGPU(double *u, double *v, int n);
void ConjugateGradient(double *A, int A_m, int A_n, double *b, double *x, int max_iter, double eps);

#endif
