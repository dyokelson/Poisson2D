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

    /*---- read in sparse matrix A  ----*/
    char matrixName[MAX_FILENAME];
    strcpy(matrixName, argv[2]);
    int is_symmetric = 0;
    read_info(matrixName, &is_symmetric);

    // Read the sparse matrix and store it in row_ind, col_ind, and val,
    int ret;
    MM_typecode matcode;
    int m_s;
    int n_s;
    int nnz;
    int *row_ind;
    int *col_ind;
    double *val;
    fprintf(stdout, "Matrix file name: %s ... ", matrixName);
    ret = mm_read_mtx_crd(matrixName, &m_s, &n_s, &nnz, &row_ind, &col_ind, &val,
                          &matcode);
    check_mm_ret(ret);
    // expand sparse matrix if symmetric
    if(is_symmetric) {
        expand_symmetry(m_s, n_s, &nnz, &row_ind, &col_ind, &val);
    }
    fprintf(stdout, "Converting COO to CSR...");
    unsigned int* csr_row_ptr = NULL;
    unsigned int* csr_col_ind = NULL;
    double* csr_vals = NULL;
    convert_coo_to_csr(row_ind, col_ind, val, m_s, n_s, nnz,
                       &csr_row_ptr, &csr_col_ind, &csr_vals);
    fprintf(stdout, "done\n");

    /*---- read in dense matrix A  ----*/
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
    strcpy(bName, argv[3]);
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

    fprintf(stdout, "Conjugate Gradient - Dense\n");
    time_t start, end, time_diff;
    start = time(NULL);
    ConjugateGradient(A, n, n, b, x, 2*n, 0.00001);
    end = time(NULL);

    time_diff = difftime(end, start);
    printf("function took %.10f seconds\n", time_diff);
        
    // write out the answer to file so we can plot with JULIA
    FILE *fp = fopen("xoutput_dense.txt", "w+");
    fprintf(fp, "%i\n", b_size);
    for (int i = 0; i < b_size; i++) {
        fprintf(fp, "%0.10lf\n", x[i]); 
    }
    fclose(fp);

    fprintf(stdout, "Conjugate Gradient - Sparse (CSR)\n");
    start = time(NULL);
    // TODO update here with sparse CG function call -
    //ConjugateGradientSparse(row_ind, col_ind, val, m_s, n_s, nnz, &csr_row_ptr, &csr_col_ind, &csr_vals);
    end = time(NULL);

    time_diff = difftime(end, start);
    printf("function took %.10f seconds\n", time_diff);

    // write out the answer to file so we can plot with JULIA
    FILE *fp2 = fopen("xoutput_sparse.txt", "w+");
    fprintf(fp2, "%i\n", b_size);
    for (int i = 0; i < b_size; i++) {
        fprintf(fp2, "%0.10lf\n", x[i]);
    }
    fclose(fp2);

}

struct Arrayz {
    int row; 
    int column;
    double value;
};

//int SortNums(struct Arrayz *a1, struct Arrayz *a2) {
int SortNums(const void *a1v, const void*a2v) { 
    struct Arrayz *a1 = (struct Arrayz*)a1v;
    struct Arrayz *a2 = (struct Arrayz*)a2v; 
    int res = a1->row - a2->row;
    return res;
}

/* This function converts a sparse matrix stored in COO format to CSR format.
   input parameters:
       int*	row_ind		list or row indices (per non-zero)
       int*	col_ind		list or col indices (per non-zero)
       double*	val		list or values  (per non-zero)
       int	m		# of rows
       int	n		# of columns
       int	n		# of non-zeros
   output parameters:
       unsigned int** 	csr_row_ptr	pointer to row pointers (per row)
       unsigned int** 	csr_col_ind	pointer to column indices (per non-zero)
       double** 	csr_vals	pointer to values (per non-zero)
   return paramters:
       none
 */
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val,
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals)

{
    // SORT
    struct Arrayz *sortedArrays;
    int size = sizeof(struct Arrayz)*nnz;
    sortedArrays = (Arrayz *) malloc(size);
    for (int i = 0; i < nnz; i++) {
        sortedArrays[i].row = row_ind[i];
        sortedArrays[i].column = col_ind[i];
        sortedArrays[i].value = val[i];
    }
    qsort(sortedArrays, nnz, sizeof(struct Arrayz), SortNums);

    // convert the rows
    *csr_row_ptr = (unsigned int *) malloc(sizeof(int) * (m+1));
    int prev_row, cur_row, row_count = 0;
    for (int k = 0; k < n; k++) {
        *(*csr_row_ptr + k) = row_count;
        while (sortedArrays[row_count].row-1 == k) {
            row_count++;
        }
    }
    // copy everything else over
    *csr_col_ind = (unsigned int *) malloc(sizeof(int) * nnz);
    *csr_vals = (double *) malloc(sizeof(double) * nnz);

    for (int x = 0; x < nnz; x++) {
        *(*csr_col_ind + x) = sortedArrays[x].column;
        *(*csr_vals + x) = sortedArrays[x].value;
    }
}


/*
 * Below functions are some matrix I/O stuff borrowed from Jee Choi's homeworks for ease of loading sparse matrices
 *
 *
 *
 * */
void usage(int argc, char** argv)
{
    if(argc < 3) {
        fprintf(stderr, "usage: %s <A dense matrix> <A sparse matrix> <b vector>\n",
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


void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind,
                     double** val)
{
    fprintf(stdout, "Expanding symmetric matrix ... ");
    int nnz = *nnz_;

    // first, count off-diagonal non-zeros
    int not_diag = 0;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            not_diag++;
        }
    }

    int* _row_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    int* _col_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    double* _val = (double*) malloc(sizeof(double) * (nnz + not_diag));

    memcpy(_row_ind, *row_ind, sizeof(int) * nnz);
    memcpy(_col_ind, *col_ind, sizeof(int) * nnz);
    memcpy(_val, *val, sizeof(double) * nnz);
    int index = nnz;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            _row_ind[index] = (*col_ind)[i];
            _col_ind[index] = (*row_ind)[i];
            _val[index] = (*val)[i];
            index++;
        }
    }

    free(*row_ind);
    free(*col_ind);
    free(*val);

    *row_ind = _row_ind;
    *col_ind = _col_ind;
    *val = _val;
    *nnz_ = nnz + not_diag;

    fprintf(stdout, "done\n");
    fprintf(stdout, "  Total # of non-zeros is %d\n", nnz + not_diag);
}

void check_mm_ret(int ret)
{
    switch(ret)
    {
        case MM_COULD_NOT_READ_FILE:
            fprintf(stderr, "Error reading file.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_PREMATURE_EOF:
            fprintf(stderr, "Premature EOF (not enough values in a line).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NOT_MTX:
            fprintf(stderr, "Not Matrix Market format.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NO_HEADER:
            fprintf(stderr, "No header information.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_UNSUPPORTED_TYPE:
            fprintf(stderr, "Unsupported type (not a matrix).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_LINE_TOO_LONG:
            fprintf(stderr, "Too many values in a line.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_COULD_NOT_WRITE_FILE:
            fprintf(stderr, "Error writing to a file.\n");
            exit(EXIT_FAILURE);
            break;
        case 0:
            fprintf(stdout, "file loaded.\n");
            break;
        default:
            fprintf(stdout, "Error - should not be here.\n");
            exit(EXIT_FAILURE);
            break;

    }
}

/* This function reads information about a sparse matrix using the
   mm_read_banner() function and printsout information using the
   print_matrix_info() function.
   input parameters:
       char*       fileName    name of the sparse matrix file
   return paramters:
       none
 */
void read_info(char* fileName, int* is_sym)
{
    FILE* fp;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;

    if((fp = fopen(fileName, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if(mm_read_banner(fp, &matcode) != 0)
    {
        fprintf(stderr, "Error processing Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    }

    if(mm_read_mtx_crd_size(fp, &m, &n, &nnz) != 0) {
        fprintf(stderr, "Error reading size.\n");
        exit(EXIT_FAILURE);
    }

    *is_sym = mm_is_symmetric(matcode);

    fclose(fp);
}

