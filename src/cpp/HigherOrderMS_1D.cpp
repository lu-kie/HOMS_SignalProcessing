#include "HigherOrderMS_1D.h"
#include <limits>

namespace HOMS
{
	/// @brief Compute the convolution of two vectors x,y
	/// @param x 
	/// @param y 
	/// @return convolution of x and y
	Eigen::VectorXd convolveVectors(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
	{
		const auto sizeX = x.size();
		const auto sizeY = y.size();
		Eigen::VectorXd conv = Eigen::VectorXd::Zero(sizeX + sizeY - 1);
		for (int j = 0; j < sizeY; j++)
		{
			for (int i = 0; i < sizeX; i++)
			{
				conv(j + i) += y(j) * x(i);
			}
		}
		return conv;

	}

	/// @brief 
	/// @param n 
	/// @param k 
	/// @param beta 
	/// @return 
	Eigen::MatrixXd computeSystemMatrix(const int n, const int k, const double beta)
	{
		if (beta < std::numeric_limits<double>::infinity())
		{
			/* Example for k = 2, beta = 1
			A = [ 1  0  0
				  ...
				  1  0  0
				  1 -2  1
				  ...
				  1 -2  1 ]
			*/
			Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n - k, k + 1);
			// The upper block of A is the identity matrix
			A.col(0).setOnes();
			// The lower block has rows given by k-fold convolutions of the k-th order finite difference vector with itself
			//Eigen::VectorXd D(-1, 1);
			Eigen::Vector2d forwardDifferenceCoeffs(-1, 1);
			Eigen::VectorXd kFoldFiniteDifferenceCoeffs(2);
			kFoldFiniteDifferenceCoeffs << -1, 1;
			for (int t = 0; t < k - 1; t++)
			{
				const auto convolutedCoeffs = convolveVectors(kFoldFiniteDifferenceCoeffs, forwardDifferenceCoeffs);
				kFoldFiniteDifferenceCoeffs.resizeLike(convolutedCoeffs);
				kFoldFiniteDifferenceCoeffs = convolutedCoeffs;
			}

			kFoldFiniteDifferenceCoeffs *= pow(beta, k);
			for (int r = n; r < 2 * n - k; r++)
			{
				A.row(r) = kFoldFiniteDifferenceCoeffs;
			}
			return A;
		}
		else
		{
			/* Example for k = 3, beta = 1
			A = [ 1    1  1
				  4    2  1
				  9    3  1
				  ...
				  n^2  n  1 ]
			*/
			Eigen::MatrixXd A = Eigen::MatrixXd::Ones(n, k);
			for (int j = k - 2; j >= 0; j--)
			{
				A.col(j) = Eigen::VectorXd::LinSpaced(n, 1, n).cwiseProduct(A.col(j + 1));
			}

			return A;
		}
	}

	/// @brief 
	/// @param n 
	/// @param k 
	/// @param beta 
	/// @param C 
	/// @param S 
	std::pair<Eigen::MatrixXd, Eigen::MatrixXd> computeGivensAngles(const int n, const int k, const double beta, mat& C, mat& S)
	{
		C.zeros();
		S.zeros();
		// Declare sparse system matrix A
		Eigen::MatrixXd A = computeSystemMatrix(n, k, beta);
		// Flag for Potts case (piecewise polynomial smoothing)
		bool potts = isinf(beta);
		// Declare aux variables
		double c, s, rho; // Givens coefficients
		int v, vv, tt, q, w, ww, off = 0; // Offset variables for the sparse matrices
		// Compute Givens coefficients to obtain QR decomposition of the system matrix
		// and save them in C,S
		for (int i = 0; i < 2 * n - k; i++) {
			if ((!potts && i < n) || (potts && i >= n)) {
				continue;
			}
			for (int j = 0; j < k + 1; j++) {
				if (potts && (j == k || i <= j)) {
					break;
				}
				// A(v,vv): Pivot element to eliminate A(i,j)
				// C(q,j),S(q,j) : Locations to store Givens coefficients
				if (potts) {
					q = i;
					v = j;
					vv = j;
					tt = k - 1;
					w = 0;
					ww = k - 1;
				}
				else {
					q = i - n + k;
					v = j + off;
					vv = 0;
					tt = k - j;
					w = j;
					ww = k;
				}
				// Determine Givens coefficients for eliminating A(i,j)
				rho = sqrt(A(v, vv) * A(v, vv) + A(i, j) * A(i, j));
				rho *= (A(v, vv) > 0) ? 1 : -1;
				c = A(v, vv) / rho;
				s = A(i, j) / rho;
				// Save the coefficients
				C(q, j) = c;
				S(q, j) = s;
				// Update A (incorporating its sparse declaration)
				rowvec A_j = A.submat(v, 0, v, tt);
				rowvec A_r = A.submat(i, w, i, ww);
				A.submat(v, 0, v, tt) = c * A_j + s * A_r;
				A.submat(i, w, i, ww) = -s * A_j + c * A_r;
			}
			if (!potts) {
				off++; // update offset aux variable
			}
		}

	}
}