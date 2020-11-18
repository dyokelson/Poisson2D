#include <cublas_v2.h>
#include <iostream>
#include "gpu_operations.h"

using namespace std;

/*
	This file contains the following functions:

		MatrixMatrixMultGPU => C = A * B
		MatrixVectorMultGPU => y = A * x
		VectorAddGPU	    => w = u + v
		VectorDotGPU	    => c = u * v
*/

void MatrixMatrixMultGPU(float *A, int A_m, int A_n, float *B, int B_m, int B_n, float *C) {
/*
	This function computes:

		C = A * B		

	MatrixMatrixMultGPU takes in 7 parameters:
		A   - matrix A
		A_m - # of rows in A
		A_n - # of columns in A
		B   - matrix B
		B_m - # of rows in B
		B_n - # of columns in B
		C   - matrix C 	
*/	

	if (A_n != B_m) {
		cout << "Matrix/Matrix sizing error" << endl;
		C = NULL;
		return;
	}
	
	int A_size = A_m * A_n;
	int B_size = B_m * B_n;
	int C_size = A_m * B_n;
	float *d_A, *d_B, *d_C;

	cudaMalloc(&d_A, A_size * sizeof(float));
	cudaMalloc(&d_B, B_size * sizeof(float));
	cudaMalloc(&d_C, C_size * sizeof(float));

	cudaMemcpy(d_A, A, A_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, B_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, C_size * sizeof(float), cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
			A_m, B_n, A_n, &alpha, d_A, 
			A_m, d_B, A_n, &beta , d_C, A_n);

	cublasDestroy(handle);
	
	cudaMemcpy(C, d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	
}

void MatrixVectorMultGPU(float *A, int A_m, int A_n, float *x, int x_m, float *y) {
/* 
	This function computes:
		
		y = Ax

	MatrixVectorMultGPU takes in 6 parameters:
		A   - matrix A
		A_m - # of rows in A
		A_n - # of columns in A
		x   - vector x
		x_n - # of elements in x
		y   - vector y			
*/

	if (A_n != x_m) {
		cout << "Matrix/Vector sizing error" << endl;
		y = NULL;
		return;
	}

	int A_size = A_m * A_n;
	float *d_A, *d_x, *d_y;
	
	cudaMalloc(&d_A, A_size * sizeof(float));
	cudaMalloc(&d_x, x_m 	* sizeof(float));
	cudaMalloc(&d_y, A_m	* sizeof(float));	

	cudaMemcpy(d_A, A, A_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, x_m    * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, x_m	  * sizeof(float), cudaMemcpyHostToDevice);
	
	cublasHandle_t handle;
	cublasCreate(&handle);

	const float alpha = 1.0f;
	const float beta = 0.0f;	
	cublasSgemv(handle, CUBLAS_OP_T, A_m, A_n, &alpha, d_A, A_m, d_x, 1, &beta, d_y, 1);
	
	cudaMemcpy(y, d_y, A_m * sizeof(float), cudaMemcpyDeviceToHost);
	
	cublasDestroy(handle);

	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_y);
}

__global__ void VectAdd(float *u, float *v, float *w, int n) {
	int i = threadIdx.x;
	if (i < n) {
		w[i] = u[i] + v[i];
	}
}

void VectorAddGPU(float *u, float *v, float *w, int n) {
/*
	This function computes:

		w = u + v

	VectorAddGPU takes in 4 parameters:
		u - vector u
		v - vector v
		w - vector w
		n - # of elements in u, v, w
*/	
	float memsize = n * sizeof(float);
	float *d_u, *d_v, *d_w;
	cudaMalloc(&d_u, memsize);
	cudaMalloc(&d_v, memsize);
	cudaMalloc(&d_w, memsize);

	cudaMemcpy(d_u, u, memsize, cudaMemcpyHostToDevice);	
	cudaMemcpy(d_v, v, memsize, cudaMemcpyHostToDevice);	
	cudaMemcpy(d_w, w, memsize, cudaMemcpyHostToDevice);	

	VectAdd<<<1, n>>>(d_u, d_v, d_w, n);

	cudaMemcpy(w, d_w, memsize, cudaMemcpyDeviceToHost);

	cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_w);
}

void VectorDotGPU(float *u, float *v, float *c, int n) {
/*
	This function computes:
		
		c = u * v

	VectorDotGPU takes 4 parameters:
		u - vector u
		v - vector v
		c - scalar output
		n - # of elements in u, v

	Usage:
		VectorDotGPU(h_u, h_v, &h_c, n);
*/
	float memsize = n * sizeof(float);
	float *d_u, *d_v, *d_c;

	cudaMalloc(&d_u, memsize);
	cudaMalloc(&d_v, memsize);
	cudaMalloc(&d_c, sizeof(float));

	cudaMemcpy(d_u, u, memsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, memsize, cudaMemcpyHostToDevice);	

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	cublasSdot(handle, n, 
			d_u, 1, 
			d_v, 1, 
			d_c);

	cublasDestroy(handle);

	cudaMemcpy(c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_c);
}

