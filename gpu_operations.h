#ifndef GPU_OPERATIONS_H
#define GPU_OPERATIONS_H

#include <cublas_v2.h>
#include <iostream>

void MatrixMatrixMultGPU(float *A, int A_m, int A_n,
			 float *B, int B_m, int B_n,
			 float *C);

void MatrixVectorMultGPU(float *A, int A_m, int A_n,
			 float *x, int x_m,
			 float *y);

void VectorAddGPU(float *u, float *v, float *w, int n);
void VectorDotGPU(float *u, float *v, float *c, int n);

#endif
