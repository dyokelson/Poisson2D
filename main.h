//
// Created by dewimaharani on 11/30/20.
//

#ifndef POISSON2D_MAIN_H
#define POISSON2D_MAIN_H
extern "C" {
    #include "mmio.h"
    #include "common.h"
}
void usage(int argc, char* argv[]);
void read_vector(char* fileName, double **vector, int *vecSize);
void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind,
                     double** val);
void check_mm_ret(int ret);
void read_info(char* fileName, int* is_sym);
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val,
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals);
void print_time(double timer[]);
#endif //POISSON2D_MAIN_H

