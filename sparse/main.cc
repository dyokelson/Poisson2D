#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
#include "gpu_sparse_operations.h"
#include "main.h"

#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100

using namespace std;

int main(int argc, char* argv[]) {

    /*---- check inputs ----*/
    usage(argc, argv);

    /*---- start timer ----*/
    double timer = 0.0;
    uint64_t t0;
    InitTSC();

    /*---- read in matrix A  ----*/
    char AName[MAX_FILENAME];
    int n;
    strcpy(AName, argv[1]);
    fprintf(stdout, "Vector file name: %s ...\n", AName);
    struct Matrix* A = read_matrix(AName);

    fprintf(stdout, "A loaded\n");
    n = A->num_cols;

    /*---- read in input vector b ----*/
    char bName[MAX_FILENAME];
    double* b;
    int b_size;
    strcpy(bName, argv[2]);
    fprintf(stdout, "Vector file name: %s ...\n ", bName);
    read_vector(bName, &b, &b_size);
    fprintf(stdout, "b loaded\n");

    /*---- read in exact answer vector ans ----*/
/*    char ansName[MAX_FILENAME];
    double* ans;
    int ans_size;
    strcpy(ansName, argv[3]);
    fprintf(stdout, "Vector file name: %s ... ", ansName);
    read_vector(ansName, &ans, &ans_size);
    fprintf(stdout, "answer loaded\n");
*/
    /*---- create vector for initial guess x (start with 0) ----*/
    fprintf(stdout, "Creating output vector...");
    srand(time(NULL));
    double *x = (double*) malloc(b_size * sizeof(double)); 
    for (int k = 0; k < b_size; k++) {
        x[k]  = rand() % 100;
    }
    fprintf(stdout, "x vector created\n");

    /*---- time the conjugate gradient ----*/    
    t0 = ReadTSC();
    ConjugateGradient(A, n, n, b, x, 2*n, 0.000000001);
    timer += ElapsedTime(ReadTSC() - t0);
    
    /*---- write output vector ----*/
    FILE *fp = fopen("xoutput.txt", "w+");
    fprintf(fp, "%i\n", b_size);
    for (int i = 0; i < b_size; i++) {
        fprintf(fp, "%0.10lf\n", x[i]); 
    }
    fclose(fp);

    /*---- print time ----*/
    print_time(timer);
}

void print_time(double time) {
    printf("Sparse Matrix Time: %0.10f\n", time);
}

void usage(int argc, char** argv)
{
    if(argc < 3) {
        fprintf(stderr, "usage: %s <A matrix> <b vector> <ans vector>\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }
}

struct Matrix *read_matrix(char* fileName) {
    int m, n, nnz;
    
    FILE* fp = fopen(fileName, "r");
    char line[MAX_NUM_LENGTH];
    fgets(line, MAX_NUM_LENGTH, fp);
    
    sscanf(line, "%d %d %d", &m, &n, &nnz);

    int *rowptr = (int *) malloc(sizeof(int) * nnz);
    int *colind = (int *) malloc(sizeof(int) * m + 1);
    double *vals = (double *) malloc(sizeof(double) * nnz);

    int i = 0;
    while (fgets(line, MAX_NUM_LENGTH, fp) != NULL) {
        int row, col;
        double val;
        
        if (i <= m) {
            sscanf(line, "%lf %d %d", &val, &row, &col);
            vals[i] = val;
            rowptr[i] = row;
            colind[i] = col;

        } else {
            sscanf(line, "%lf %d", &val, &row);
            vals[i] = val;
            rowptr[i] = row;

        }
        i++;
    }
    fclose(fp);
    
    struct Matrix *mat = (struct Matrix *) malloc(sizeof(struct Matrix));

    mat->num_rows = m;
    mat->num_cols = n;
    mat->nnz = nnz;

    mat->h_rowptr = rowptr;
    mat->h_vals = vals;
    mat->h_colind = colind;

    return mat;
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
    printf("index: %i, vector_size: %i\n", index, vector_size);

    *vector = vector_;
    *vecSize = vector_size;
}
