#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
#include "gpu_tests.h" 
#include "gpu_operations.h"
#include "main.h"

#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100

using namespace std;

int main(int argc, char* argv[]) {

    /*---- check inputs ----*/
    usage(argc, argv);

    /*---- test gpu functions ----*/
    //test_gpu_operations();

    /*---- read in matrix A  ----*/
    char AName[MAX_FILENAME];
    double* A;
    int A_dim, n;
    strcpy(AName, argv[1]);
    fprintf(stdout, "Vector file name: %s ...\n ", AName);
    read_vector(AName, &A, &A_dim);
    fprintf(stdout, "A loaded\n");
    n = sqrt(A_dim);

    /*---- read in input vector b ----*/
    char bName[MAX_FILENAME];
    double* b;
    int b_size;
    strcpy(bName, argv[2]);
    fprintf(stdout, "Vector file name: %s ...\n ", bName);
    read_vector(bName, &b, &b_size);
    fprintf(stdout, "b loaded\n");

    /*---- create vector for initial guess x (start with 0) ----*/
    fprintf(stdout, "Creating output vector...");
    srand(time(NULL));
    double *x = (double*) malloc(b_size * sizeof(double)); 
    for (int k = 0; k < b_size; k++) {
        x[k] = rand() % 100;
    }
    fprintf(stdout, "x vector created\n");
    
    time_t start, end, time_diff;
    start = time(NULL);
    ConjugateGradient(A, n, n, b, x, 2*n, 0.00001);
    end = time(NULL);

    time_diff = difftime(end, start);
    printf("function took %.10f seconds\n", time_diff);
        
    // write out the answer to file so we can plot with JULIA
    //ostringstream outfilename; 
    //outfilename << "xoutput_" << AName;
    FILE *fp = fopen("xoutput.txt", "w+");
    fprintf(fp, "%i\n", b_size);
    for (int i = 0; i < b_size; i++) {
        fprintf(fp, "%0.10lf\n", x[i]); 
    }
    fclose(fp);

}


void usage(int argc, char** argv)
{
    if(argc < 2) {
        fprintf(stderr, "usage: %s <A matrix> <b vector>\n",
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
    printf("index: %i, vector_size: %i\n", index, vector_size);

    *vector = vector_;
    *vecSize = vector_size;
}
