/**
    ArmadilloConverter.cpp
    Purpose: Converts (MATLAB) pointers to armadillo objects and calls the C++ functions
             Called by the mexFunction HigherOrderMS1D_wrapper from MATLAB

    @author Lukas Kiefer
    @version 1.0
*/
#include "HigherOrderMS_1D.h"

void armadilloConverter(double* data_raw, const int n, const int k, const double gamma,
						const double beta, double* J_raw, double* u_raw)
{
	// Create Armadillo objects from raw pointers (MATLAB arrays)
    // The used constructors make sure that the memory corresponding to the pointers is used
    vec data = vec(data_raw, n, false,  true);
    vec u = vec(u_raw, n, false,  true);
	vec J = vec(J_raw, n, false,  true);
    // Determine the recursion coefficients given by Givens rotation angles/coefficients
	mat C,S;
	if (std::isinf(beta)){
		C = zeros(n,k), S=zeros(n,k);
	} else{
		C = zeros(n,k+1), S=zeros(n,k+1);
	}
    calcGivensAngles(n,k,beta,C,S);
    // Compute all [1,r]-errors (general approx. errors are computed on the fly in findBestPartition)
    vec eps1R = compute1rErrors(data,n,k,beta,C,S);
    // Determine an optimal partitioning encoded by J
    findBestPartition(data,n,gamma,eps1R,k,beta,C,S,J);
    // Reconstruct the optimal smoothed signal u for the optimal partitioning J
	reconstructionFromPartition(J,u,data,n,k,beta,C,S);
}
