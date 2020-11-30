#include <iostream>
#include "gpu_operations.h"
#include "cg.h"
#include "main.h"

#define M 4
#define N 4
#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100

using namespace std;

int main(int argc, char* argv[]) {

    /*---- check inputs ----*/
    usage(argc, argv);

    /*---- test gpu functions ----*/
    test_gpu_operations();

    /*---- read in matrix A  ----*/
    char AName[MAX_FILENAME];
    strcpy(AName, argv[1]);
    fprintf(stdout, "Vector file name: %s ... ", AName);
    double* A;
    int vector_size;
    read_vector(AName, &A, &vector_size);
    fprintf(stdout, "A loaded\n");

    /*---- read in input vector b ----*/
    char bName[MAX_FILENAME];
    strcpy(bName, argv[2]);
    fprintf(stdout, "Vector file name: %s ... ", bName);
    double* b;
    int b_size;
    read_vector(bName, &b, &b_size);
    fprintf(stdout, "b loaded\n");

    /*---- read in exact answer vector ans ----*/
    char ansName[MAX_FILENAME];
    strcpy(ansName, argv[3]);
    fprintf(stdout, "Vector file name: %s ... ", ansName);
    double* ans;
    int ans_size;
    read_vector(ansName, &ans, &ans_size);
    fprintf(stdout, "answer loaded\n");

    /*---- create vector for initial guess x (start with 0) ----*/
    fprintf(stdout, "Creating output vector...");
    double *x = (double*) malloc(pow(vector_size+2,2) * sizeof(double)); // TODO: correct dims?
    memset(x, 0, vector_size);
    fprintf(stdout, "x vector created\n");

    // TODO: call cg here (once refactored)

    // TODO: write out the answer to file so we can plot with JULIA

    // TODO: additional problems? like non-zero boundary conditions?

}

void test_gpu_operations() {
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
    VectorAddGPU(u, v, -1.0, w, N);
    for (int i = 0; i < N; i++) {
        cout << w[i] << endl;
    }

    cout << endl << "Vector Dot:" << endl;
    c = VectorDotGPU(u, v, N);
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
}

void usage(int argc, char** argv)
{
    if(argc < 3) {
        fprintf(stderr, "usage: %s <A matrix> <b vector> <ans vector>\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
}

void read_vector(char* fileName, double** vector, int* vecSize)
{
    FILE* fp = fopen(fileName, "r");
    char line[MAX_NUM_LENGTH];
    fgets(line, MAX_NUM_LENGTH, fp);
    fclose(fp);

    unsigned int vector_size = atoi(line);
    double* vector_ = (double*) malloc(sizeof(double) * vector_size);

    fp = fopen(fileName, "r");
    // first read the first line to get the # elements
    fgets(line, MAX_NUM_LENGTH, fp);

    unsigned int index = 0;
    while(fgets(line, MAX_NUM_LENGTH, fp) != NULL) {
        vector_[index] = atof(line);
        index++;
    }

    fclose(fp);
    printf("index: %i, vector_size: %i", index, vector_size);

    *vector = vector_;
    *vecSize = vector_size;
}
