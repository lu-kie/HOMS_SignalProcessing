/**
	CalcGivensAngles.cpp
	Purpose: Computes the recurrence coefficients needed by the dynamic programming solver for the
			 1D k-th order Mumford-Shah/Potts model. It performs effectively a QR
			 decomposition of the system matrix of a least squares problem and saves the
			 necessary rotation angles to obtain the upper triangluar matrix R

	@author Lukas Kiefer
	@version 1.0
*/

#include "HigherOrderMS_1D.h"


void calcGivensAngles(const int n, const int k, const double beta, mat &C, mat &S)
{
	C.zeros();
    S.zeros();
    // Declare sparse system matrix A
	mat A = constructSystemMatrix(n,k,beta);
	// Flag for Potts case (piecewise polynomial smoothing)
	bool potts = isinf(beta);
	// Declare aux variables
    double c,s,rho; // Givens coefficients
	int v,vv,tt,q,w,ww,off = 0; // Offset variables for the sparse matrices
	// Compute Givens coefficients to obtain QR decomposition of the system matrix
	// and save them in C,S 
    for(int i = 0; i < 2*n-k; i++) {
		if ((!potts && i < n) || (potts && i >= n)){
			continue;
		}
		for(int j = 0; j < k+1; j++) {
			if (potts && (j == k || i <= j)){
				break;
			}
			// A(v,vv): Pivot element to eliminate A(i,j)
			// C(q,j),S(q,j) : Locations to store Givens coefficients
			if (potts){
				q  = i;
				v  = j;
				vv = j;
				tt = k-1;
				w  = 0;
				ww = k-1;
			} else {
				q  = i-n+k;
				v  = j+off;
				vv = 0;
				tt = k-j;
				w  = j;
				ww = k;
			}
            // Determine Givens coefficients for eliminating A(i,j)
            rho = sqrt(A(v,vv)*A(v,vv) + A(i,j)*A(i,j));
            rho *= (A(v,vv) > 0) ? 1 : -1;
            c = A(v,vv) / rho;
            s = A(i,j) / rho;
			// Save the coefficients 
            C(q,j) = c;
            S(q,j) = s;
			// Update A (incorporating its sparse declaration)
            rowvec A_j = A.submat(v,0,v,tt);
            rowvec A_r = A.submat(i,w,i,ww);
            A.submat(v,0,v,tt) =  c*A_j+s*A_r;
            A.submat(i,w,i,ww) = -s*A_j+c*A_r;
        }
		if(!potts){
			off++; // update offset aux variable
		}
    }
}

// Aux function for declaring sparse matrix
rowvec convolution1D(rowvec x, rowvec y)
{
    int n1 = x.n_elem;
    int n2 = y.n_elem;
    rowvec conv = zeros(1,n1+n2-1);
    for(int j = 0; j < n2 ; j++) {
        for(int i = 0; i < n1 ; i++) {
            conv(j+i) +=  + y(j)*x(i);
        }
    }
    return conv;
}

// Aux function for declaring sparse matrix
mat constructSystemMatrix(const int n, const int k, const double beta){
	mat A;
	if(beta < numeric_limits<double>::infinity()){
		A = zeros(2*n-k,k+1);
		// The upper block of A is the identity matrix
		A.col(0) = ones(2*n-k);
		// The lower block has rows given by k-fold convolutions of the k-th order finite difference vector with itself
		rowvec D(2),D_h(2);
		D   << -1 << 1 << endr;
		D_h << -1 << 1 << endr;
		for(int t = 0; t < k-1 ; t++)
			D = convolution1D(D,D_h);
		D *= pow(beta,k);
		for(int r = n; r < 2*n-k; r++){
			A.row(r) = D;
		}	
	} else {
		// Create system matrix for polynomial regression
		A = zeros(n,k);
		vec q = linspace<vec>(1, n, n);
		for(int l=0; l<k; l++){
			A.col(k-l-1) = pow(q,l);
		}
	}
	return A;
}