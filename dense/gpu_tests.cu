#include "gpu_tests.h"
#include <iostream>
#include "gpu_operations.h"

#define M 4
#define N 4

using namespace std;

void test_gpu_operations() {
    double c;
    double *A = (double *)malloc(N * M * sizeof(double)); double *d_A; cudaMalloc(&d_A, N * M * sizeof(double));
    double *C = (double *)malloc(N * M * sizeof(double)); double *d_C; cudaMalloc(&d_C, N * M * sizeof(double));
    double *u = (double *)malloc(N * sizeof(double));     double *d_u; cudaMalloc(&d_u, N * sizeof(double));
    double *v = (double *)malloc(N * sizeof(double));     double *d_v; cudaMalloc(&d_v, N * sizeof(double));
    double *w = (double *)malloc(N * sizeof(double));     double *d_w; cudaMalloc(&d_w, N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));     double *d_x; cudaMalloc(&d_x, N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));     double *d_y; cudaMalloc(&d_y, N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double)); 


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[(i * N) + j] = i;
            C[(i * N) + j] = 0;
        }
        u[i] = i;
        v[i] = i;
        w[i] = 0;
        x[i] = 0;
    }
    x[3] = 1;

    cudaMemcpy(d_A, A, N * M * sizeof(double), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_u, u, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    cout << "Vector Add:" << endl;
    VectorAddGPU(d_u, d_v, 1.0, d_w, N);
    cudaMemcpy(w, d_w, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        cout << w[i] << endl;
    }

    cout << endl << "Vector Dot:" << endl;
    c = VectorDotGPU(d_u, d_v, N);
    cout << c << endl;

    cout << endl << "MatrixVector Mult:" << endl;
    MatrixVectorMultGPU(d_A, M, N, d_x, N, d_y);
    cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        cout << y[i] << endl;
    }

    cout << endl << "MatrixMatrix Mult:" << endl;
    MatrixMatrixMultGPU(d_A, M, N, d_A, M, N, d_C);
    cudaMemcpy(C, d_C, N * M * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << C[(i * N) + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);

    cout << endl << "ConjGrad:" << endl;   
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) { 
                A[(i*N) + j] = 5; 
            } else { 
                A[(i*N) + j] = 0; 
            } 
        }
    }

    for (int i = 0; i < M; i++) {
        x[i] = 0;
        b[i] = i * i;
    } 

    ConjugateGradient(A, M, N, b, x, 5, 0.01);
    for (int i = 0; i < M; i++) {
        cout << x[i] << endl;
    }


}
