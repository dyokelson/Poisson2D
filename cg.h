//
// Created by dewimaharani on 11/18/20.
//

#ifndef POISSON2D_CG_H
#define POISSON2D_CG_H

// Solving Ax = b, with initial guess x
void ConjugateGradient(float *A, int A_m, int A_n,
                         float *B, int B_m, int B_n,
                         float *x, int max_iter, double eps);
#endif //POISSON2D_CG_H
