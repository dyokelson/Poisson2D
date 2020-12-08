#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <helper_cuda.h>
#include <iostream>
#include "gpu_sparse_operations.h"

using namespace std;

/*
	This file contains the following functions:

		MatrixVectorMultGPU => y = A * x
		VectorAddGPU	    => w = u + v
		VectorDotGPU	    => c = u * v
*/

void MatrixVectorMultGPU(struct Matrix *d_A, int A_m, int A_n, double *d_x, int x_m, double *d_y) {
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

	cusparseHandle_t handle;
	cusparseCreate(&handle);

    cusparseMatDescr_t descA;
    cusparseCreateMatDescr(&descA);

	const double alpha = 1.0f;
	const double beta = 0.0f;	

    cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, A_m, A_n, d_A->nnz, &alpha, 
                    descA, 
                    d_A->d_vals, 
                    d_A->d_colind, 
                    d_A->d_rowptr, 
                    d_x, 
                    &beta, 
                    d_y);
	
    cusparseDestroy(handle);

}

__global__ void VectAdd(double *u, double *v, double a, double *w, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
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
	
    VectAdd<<<8, 256>>>(d_u, d_v, a, d_w, n);

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
	checkCudaErrors(cudaMalloc(&d_c, sizeof(double)));

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

	cublasDdot(handle, n, 
			d_u, 1, 
			d_v, 1, 
			d_c);

	cublasDestroy(handle);

	checkCudaErrors(cudaMemcpy(c, d_c, sizeof(double), cudaMemcpyDeviceToHost));

	return *c;
}

void ConjugateGradient(struct Matrix *A, int A_m, int A_n, double *b, double *x, int max_iter, double eps) {
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

    double residual_old, residual_new, d, alpha, beta;
    double *d_x, *d_b, *d_a_p, *d_r_k, *d_p_k; 

    int res_length = 0;
    double *resids = (double *)malloc(sizeof(double) * max_iter);

    checkCudaErrors(cudaMalloc(&(A->d_colind), (A_m + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&(A->d_rowptr), A->nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc(&(A->d_vals),   A->nnz * sizeof(double)));

    checkCudaErrors(cudaMemcpy(A->d_colind, A->h_colind, (A_m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(A->d_rowptr, A->h_rowptr, A->nnz    * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(A->d_vals,   A->h_vals,   A->nnz    * sizeof(double), cudaMemcpyHostToDevice));
 
    checkCudaErrors(cudaMalloc(&d_x,   A_m    * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_b,   A_m    * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_p_k, A_m    * sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_a_p, A_m    * sizeof(double)));	
    checkCudaErrors(cudaMalloc(&d_r_k, A_m    * sizeof(double)));

	checkCudaErrors(cudaMemcpy(d_x, x, A_m    * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, A_m    * sizeof(double), cudaMemcpyHostToDevice)); 

								                                                // Calculate inital residual, b-Ax with initial guess
    MatrixVectorMultGPU(A, A_m, A_n, d_x, A_m, d_a_p);	                        // ap = Ax 
    VectorAddGPU(d_b, d_a_p, -1.0, d_r_k, A_m);			                        // r = b - ap 
    residual_old = VectorDotGPU(d_r_k, d_r_k, A_m);			                    // res_o = dot(r, r)
    checkCudaErrors(cudaMemcpy(d_p_k, d_r_k, A_m * sizeof(double), cudaMemcpyDeviceToDevice));   // p = r
                                								                // Iterate until converges or max_iter
    for (int i = 0; i < max_iter; i++) {			                            // for i:max_iterations:
        MatrixVectorMultGPU(A, A_m, A_n, d_p_k, A_m, d_a_p);                    //  ap = Ap
        d = VectorDotGPU(d_p_k, d_a_p, A_m);			                        // 	d = dot(p, ap)
        alpha = residual_old / d;				                                //	alpha = res_o / d
        VectorAddGPU(d_x, d_p_k, alpha, d_x, A_m);			                    //	x = x + (alpha * p)
        VectorAddGPU(d_r_k, d_a_p, -alpha, d_r_k, A_m);		                    //	r = r - (alpha * ap)	
        residual_new = VectorDotGPU(d_r_k, d_r_k, A_m);		                    //	res_n = dot(r, r)


        //printf("Iterations: %i Residual Old: %0.10lf\n", i, sqrt(residual_old));
        //printf("Iterations: %i Residual New: %0.10lf\n", i, sqrt(residual_new));
        if (sqrt(residual_new) < eps) {				                            // if sqrt(res_n) < eps):
            printf("Converged in iterations: %i Residual: %0.10lf\n", i, sqrt(residual_new));
            break;						                                        //  exit
        }

        beta = residual_new / residual_old;			                            // beta = res_n / res_o
        VectorAddGPU(d_r_k, d_p_k, beta, d_p_k, A_m);			                // p = r + (beta * p)
        resids[res_length] = residual_old;
        res_length++;
        residual_old = residual_new;				                            // res_o = res_n

    }
    
    cudaMemcpy(x, d_x, A_m * sizeof(double), cudaMemcpyDeviceToHost);

    //FILE *fp = fopen("residuals.txt", "w+");
    //fprintf(fp, "%i\n", res_length);
    //for (int i = 0; i < res_length; i++) {
        //fprintf(fp, "%0.10lf\n", resids[i]); 
    //}
    //fclose(fp);
    
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_a_p);
    cudaFree(d_r_k);
    cudaFree(d_p_k);
}
