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

void MatrixMatrixMultGPU(double *d_A, int A_m, int A_n, double *d_B, int B_m, int B_n, double *d_C) {
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
		d_C = NULL;
		return;
	}
	
	cublasHandle_t handle;
	cublasCreate(&handle);

	const double alpha = 1.0f;
	const double beta = 0.0f;
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
			A_m, B_n, A_n, &alpha, d_A, 
			A_m, d_B, A_n, &beta , d_C, A_n);

	cublasDestroy(handle);	
}

void MatrixVectorMultGPU(double *d_A, int A_m, int A_n, double *d_x, int x_m, double *d_y) {
/* 
	This function computes:
		
		y = Ax

	MatrixVectorMultGPU takes in 6 parameters:
		A   - matrix A
		A_m - # of rows in A
		A_n - # of columns in A
		x   - vector x
		x_m - # of elements in x
		y   - vector y			
*/

	if (A_n != x_m) {
		cout << "Matrix/Vector sizing error" << endl;
		d_y = NULL;
		return;
	}

	cublasHandle_t handle;
	cublasCreate(&handle);

	const double alpha = 1.0f;
	const double beta = 0.0f;	
	cublasDgemv(handle, CUBLAS_OP_T, A_m, A_n, &alpha, d_A, A_m, d_x, 1, &beta, d_y, 1);
	
	cublasDestroy(handle);
}

__global__ void VectAdd(double *u, double *v, double a, double *w, int n) {
	int i = threadIdx.x;
	if (i < n) {
		w[i] = u[i] + (a*v[i]);
	}
}

void VectorAddGPU(double *d_u, double *d_v, double a, double *d_w, int n) {
/*
	This function computes:

		w = u + (a*v)

	VectorAddGPU takes in 5 parameters:
		u - vector u
		v - vector v
		a - double a
		w - vector w
		n - # of elements in u, v, w
*/	
	
    VectAdd<<<1, n>>>(d_u, d_v, a, d_w, n);

}

double VectorDotGPU(double *d_u, double *d_v, int n) {
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
	
    double *d_c;
    double *c = (double *)malloc(sizeof(double));
	cudaMalloc(&d_c, sizeof(double));

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	cublasDdot(handle, n, 
			d_u, 1, 
			d_v, 1, 
			d_c);

	cublasDestroy(handle);

	cudaMemcpy(c, d_c, sizeof(double), cudaMemcpyDeviceToHost);

	return *c;
}

void ConjugateGradient(double *A, int A_m, int A_n, double *b, double *x, int max_iter, double eps) {
/*
	This function computes:

		Ax = b

	ConjugateGradient takes in 6 parameters:
		A - matrix A
		A_m - number of rows in A (assuming square matrix)
		b - vector b
		x - vector x, an initial guess
		max_iter - maximum number of times to iterate
		eps - the tolerance, or very small number that will tell us if it has converged
*/

	int A_size = A_m * A_n;
    double residual_old, residual_new, d, alpha, beta;
    double *d_A, *d_x, *d_b, *d_a_p, *d_r_k, *d_p_k; 

	cudaMalloc(&d_A,   A_size * sizeof(double));
	cudaMalloc(&d_x,   A_m    * sizeof(double));
    cudaMalloc(&d_b,   A_m    * sizeof(double));
	cudaMalloc(&d_a_p, A_m    * sizeof(double));	
    cudaMalloc(&d_r_k, A_m    * sizeof(double));
    cudaMalloc(&d_p_k, A_m    * sizeof(double));

    cudaMemcpy(d_A, A, A_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, A_m    * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, A_m    * sizeof(double), cudaMemcpyHostToDevice);

								                                                // Calculate inital residual, b-Ax with initial guess
    MatrixVectorMultGPU(d_A, A_m, A_n, d_x, A_m, d_a_p);	                    // a = Ax 
    VectorAddGPU(d_b, d_a_p, -1.0, d_r_k, A_m);			                        // r = b - a 
    residual_old = VectorDotGPU(d_r_k, d_r_k, A_m);			                    // res_o = dot(r, r)
    cudaMemcpy(d_p_k, d_r_k, A_m * sizeof(double), cudaMemcpyDeviceToDevice);   // p = r

                                								                // Iterate until converges or max_iter
    for (int i = 0; i < max_iter; i++) {			                            // for i:max_iterations:
        MatrixVectorMultGPU(d_A, A_m, A_n, d_p_k, A_m, d_a_p);                  // 	a = Ap
        d = VectorDotGPU(d_p_k, d_a_p, A_m);			                        // 	d = dot(p, a)
        alpha = residual_old / d;				                                //	alpha = res_o / d
        VectorAddGPU(d_x, d_p_k, -alpha, d_x, A_m);			                    //	x = x + (alpha * p)
        VectorAddGPU(d_r_k, d_a_p, -alpha, d_r_k, A_m);		                    //	r = r - (alpha * a)	
        residual_new = VectorDotGPU(d_r_k, d_r_k, A_m);		                    //	res_n = dot(r, r)

		       						                                            // Check for convergence
        if (sqrt(residual_new) < eps) {				                            // if sqrt(res_n) < eps):
            printf("Converged before max iterations! Residual: %f\n", sqrt(residual_new));
            break;						                                        //  exit
        }

        beta = residual_new / residual_old;			                            // beta = res_n / res_o
        VectorAddGPU(d_r_k, d_p_k, beta, d_p_k, A_m);			                // p = r + (beta * p)
        residual_old = residual_new;				                            // res_o = res_n

    }
    
    cudaMemcpy(x, d_x, A_m * sizeof(double), cudaMemcpyDeviceToHost);

    printf("X Vector:\n");
    for (int k = 0; k < A_n; k++) {
        printf("%f\n", x[k]);
    } 
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_a_p);
    cudaFree(d_r_k);
    cudaFree(d_p_k);
}
