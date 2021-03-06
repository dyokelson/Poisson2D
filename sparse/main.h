#ifndef POISSON2D_MAIN_H
#define POISSON2D_MAIN_H

extern "C" {
    #include "common.h"
}

void usage(int argc, char* argv[]);
void print_time(double time);
void read_vector(char* fileName, double  **A, int *vecSize);
struct Matrix *read_matrix(char* fileName);

#endif //POISSON2D_MAIN_H

