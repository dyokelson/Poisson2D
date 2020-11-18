#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gpu_operations.h"



void ConjugateGradient(double *A, int A_m, double *B, double *x, int max_iter, double eps) {
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

    // Setup
    double residual_old, residual_new = 0.0;
    double r_k = (double*) malloc(sizeof(double) * A_m);
    double p_k = (double*) malloc(sizeof(double) * A_m);
    double a_p = (double*) malloc(sizeof(double) * A_m);

    // Calculate inital residual, b-Ax with initial guess
    MatrixVectorMultGPU(A, A_m, A_n, x, A_m, a_p);
    VectorAddGPU(b, a_p, -1.0, r_k, A_m);
    VectorDotGPU(r_k, r_k, residual_old, A_m);

    memcpy(p_k, r_k, sizeof(dobule)* A_m);

    double d, alpha, beta = 0.0;
    // Iterate until converges or max_iter
    for (int i = 0; i < max_iter; i++) {
        // A*p
        MatrixVectorMultGPU(A, A_m, A_n, p_k, A_m, a_p);
        // d = p^t * A * p
        VectorDotGPU(p_k, a_p, d, A_m);
        // alpha = residual / d
        alpha = residual_old / d;
        // x_k+1 = x_k + alpha * p_k
        VectorAddGPU(x_k, p_k, alpha, x_k, A_m);
        // r_k+1 = r_k - alpha * A * p
        VectorAddGPU(r_k, a_p, -alpha, r_k, A_m);
        // calculate new residual
        VectorDotGPU(r_k, r_k, residual_new, A_m);

        // check for convergence
        if (sqrt(residual_new) < eps) {
            break;
        }

        // beta = residual_new / residual_old
        beta = residual_new / residual_old;
        // p_k+1 = r_k+1 + beta * p_k
        VectorAddGPU(r_k, p_k, beta, p_k, A_m);
        // update residual
        residual_old = residual_new;

    }
}