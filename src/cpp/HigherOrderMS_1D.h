#ifndef HIGHERORDERMS1D_H
#define HIGHERORDERMS1D_H

#include <list>
#include <iterator>
#include <math.h>
#define ARMA_NO_DEBUG
#include <armadillo>
#include "Interval.h"

using namespace std;
using namespace arma;

// Converter from pointers to armadillo objects
void armadilloConverter(double* data_raw, const int n, const int k, const double gamma,
                        const double beta, double* J_raw, double* u_raw);
// Computes the recursion coefficients for the dynamic programming scheme
void calcGivensAngles(const int n, const int k,const double beta ,mat &C, mat &S);
// Computes and stores the approximation errors for intervals [1,r] for all r
vec compute1rErrors(vec &data, const int n, const int k, const double beta, mat &C, mat &S);
// Computes the optimal univariate partitioning for data data
void findBestPartition(vec &data,const int n, const double gamma, vec &eps_1r,
                       const int k, const double beta, mat &C, mat &S,vec &J);
// Computes the corresponding reconstruction for an optimal partition
void reconstructionFromPartition(vec &J, vec &u, vec &data, const int n, const int k, const double beta, mat &C, mat &S);
// Aux functions for setting up the (sparse) system matrix
mat constructSystemMatrix(const int n,const int k, const double beta);
rowvec convolution1D(rowvec x, rowvec y);
// Aux function for comparing interval lengths for the reconstruction
bool compare_intervLengths(Interval inter1, Interval inter2);
#endif  
