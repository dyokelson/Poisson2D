#include "gpu_operations.h"
#include "cg.h"
#include <iostream>

#define M 4
#define N 4

using namespace std;

int main() {
	
	double c;
	double *A = (double *)malloc(N * M * sizeof(double));
	double *C = (double *)malloc(N * M * sizeof(double));
	double *u = (double *)malloc(N * sizeof(double));
	double *v = (double *)malloc(N * sizeof(double));
	double *w = (double *)malloc(N * sizeof(double));
	double *x = (double *)malloc(N * sizeof(double));
	double *y = (double *)malloc(N * sizeof(double));

	for (int i = 0; i < N; i++) {	
		for (int j = 0; j < N; j++) {
			A[(i * N) + j] = i;
		}
		u[i] = i;
		v[i] = i;
		w[i] = 0;
		x[i] = 0;
	}
	x[3] = 1;

	cout << "Vector Add:" << endl;
	VectorAddGPU(u, v, w, N);		
	for (int i = 0; i < N; i++) {
		cout << w[i] << endl;
	}

	cout << endl << "Vector Dot:" << endl;
	VectorDotGPU(u, v, &c, N);	
	cout << c << endl;

	cout << endl << "MatrixVector Mult:" << endl;
	MatrixVectorMultGPU(A, M, N, x, N, y);
	for (int i = 0; i < N; i++) {
		cout << y[i] << endl;
	}

	cout << endl << "MatrixMatrix Mult:" << endl;
	MatrixMatrixMultGPU(A, M, N, A, M, N, C);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout << C[(i * N) + j] << " ";
		} 
		cout << endl;
	}

	// TODO: Setup for CG and test/experiment here (Dewi)

}
