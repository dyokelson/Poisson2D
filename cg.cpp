#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu_operations.h"



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

    								// initilize
    double residual_old, residual_new = 0.0;
    double d, alpha, beta = 0.0;
    double *r_k = (double*)malloc(sizeof(double) * A_m);
    double *p_k = (double*)malloc(sizeof(double) * A_m);
    double *a_p = (double*)malloc(sizeof(double) * A_m);

								// Calculate inital residual, b-Ax with initial guess
    MatrixVectorMultGPU(A, A_m, A_n, x, A_m, a_p);		// a = Ax
    VectorAddGPU(b, a_p, -1.0, r_k, A_m);			// r = b - a
    residual_old = VectorDotGPU(r_k, r_k, A_m);			// res_o = dot(r, r)
    memcpy(p_k, r_k, sizeof(double)* A_m);			// p = r

								// Iterate until converges or max_iter
    for (int i = 0; i < max_iter; i++) {			// for i:max_iterations:
        MatrixVectorMultGPU(A, A_m, A_n, p_k, A_m, a_p);  	// 	a = Ap
        d = VectorDotGPU(p_k, a_p, A_m);			// 	d = dot(p, a)
        alpha = residual_old / d;				//	alpha = res_o / d
        VectorAddGPU(x, p_k, alpha, x, A_m);			//	x = x + (alpha * p)
        VectorAddGPU(r_k, a_p, -alpha, r_k, A_m);		//	r = r - (alpha * a)	
        residual_new = VectorDotGPU(r_k, r_k, A_m);		//	res_n = dot(r, r)

		       						//      check for convergence
        if (sqrt(residual_new) < eps) {				//	if sqrt(res_n) < eps):
            break;						//		exit
        }

        beta = residual_new / residual_old;			//	beta = res_n / res_o
        VectorAddGPU(r_k, p_k, beta, p_k, A_m);			//	p = r + (beta * p)
        residual_old = residual_new;				//	res_o = res_n

    }
}
