/**
    Compute1rErrors.cpp
    Purpose: Computes the approximation errors for discrete
             intervals [1,r] for all r with Givens rotations

    @author Lukas Kiefer
    @version 1.0
*/

#include "HigherOrderMS_1D.h"

vec compute1rErrors(vec &data, const int n, const int k, const double beta, mat &C, mat &S)
{
    vec eps1R = zeros(n);
    // Create an interval object to compute the approximation errors on all intervals [1,r]
    Interval err = Interval(1,1,data(0),k, beta);
    for(int r = 1; r < n; r++){
        // Compute the approximation error for interval [1,r] using the error update function of the Interval class
        err.addBottomDataPoint(k,C,S,data(r));
        eps1R(r) = err.getEps();
    }

    return eps1R;
}
