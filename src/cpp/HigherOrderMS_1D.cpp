#include "HigherOrderMS_1D.h"
#include <limits>
#include <iostream>
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
	/// @param dataLength 
	/// @param smoothnessOrder 
	/// @param smoothnessPenalty 
	/// @return 
	Eigen::MatrixXd computeSystemMatrix(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty)
	{
		if (smoothnessOrder < 1)
		{
			throw std::invalid_argument("The requested order must be at least 1");
		}

		if (smoothnessPenalty < std::numeric_limits<double>::infinity())
		{
			/* Example for k = 2, beta = 1
			A = [ 1  0  0
				  ...
				  1  0  0
				  1 -2  1
				  ...
				  1 -2  1 ]

			which is a sparse representation of the full system matrix

			[1  0  0   ... 0
			 0  1  0   ... 0
			 ...
			 0  0  0   ... 1
			 1 -2  1   ... 0
			 0  1 -2 1 ... 0
			 ...
			 ...      1 -2 1 ]
			*/

			Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * dataLength - smoothnessOrder, smoothnessOrder + 1);
			// The upper block of A is the identity matrix
			A.col(0).setOnes();

			// The lower block has rows given by k-fold convolutions of the k-th order finite difference vector with itself
			Eigen::Vector2d forwardDifferenceCoeffs(-1, 1);
			Eigen::VectorXd kFoldFiniteDifferenceCoeffs(2);
			kFoldFiniteDifferenceCoeffs << -1, 1;
			for (int t = 0; t < smoothnessOrder - 1; t++)
			{
				const auto convolutedCoeffs = convolveVectors(kFoldFiniteDifferenceCoeffs, forwardDifferenceCoeffs);
				kFoldFiniteDifferenceCoeffs.resizeLike(convolutedCoeffs);
				kFoldFiniteDifferenceCoeffs = convolutedCoeffs;
			}

			kFoldFiniteDifferenceCoeffs *= pow(smoothnessPenalty, smoothnessOrder);
			for (int r = dataLength; r < 2 * dataLength - smoothnessOrder; r++)
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
			Eigen::MatrixXd A = Eigen::MatrixXd::Ones(dataLength, smoothnessOrder);
			for (int j = smoothnessOrder - 2; j >= 0; j--)
			{
				A.col(j) = Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength).cwiseProduct(A.col(j + 1));
			}

			return A;
		}
	}

	/// @brief 
	/// @param dataLength 
	/// @param smoothnessOrder 
	/// @param smoothnessPenalty 
	GivensCoefficients::GivensCoefficients(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty)
	{
		const bool isPiecewisePolynomial = std::isinf(smoothnessPenalty);
		if (isPiecewisePolynomial)
		{
			C = Eigen::MatrixXd::Zero(dataLength, smoothnessOrder);
			S = Eigen::MatrixXd::Zero(dataLength, smoothnessOrder);
		}
		else
		{
			C = Eigen::MatrixXd::Zero(dataLength, smoothnessOrder + 1);
			S = Eigen::MatrixXd::Zero(dataLength, smoothnessOrder + 1);
		}

		auto systemMatrix = computeSystemMatrix(dataLength, smoothnessOrder, smoothnessPenalty);
		// aux variables
		double rho;
		int vv, tt, q, w, off = 0; // offsets to compensate for sparse systemMatrix

		// Compute the coefficients of the Givens rotations to compute a QR decomposition of the systemMatrix.
		// Save them in C and S
		for (int i = 0; i < systemMatrix.rows(); i++)
		{
			if (!isPiecewisePolynomial && i < dataLength)
			{
				continue;
			}

			for (int j = 0; j < smoothnessOrder + 1; j++)
			{
				if (isPiecewisePolynomial && (j == smoothnessOrder || i <= j))
				{
					break;
				}

				// systemMatrix(v, vv): Pivot element to eliminate the entry systemMatrix(i,j)
				// C(q,j), S(q,j): locations to store the corresponding Givens coefficients
				if (isPiecewisePolynomial)
				{
					q = i;
					vv = j;
					tt = smoothnessOrder;
					w = 0;
				}
				else
				{
					q = i - dataLength + smoothnessOrder;
					vv = 0;
					tt = smoothnessOrder - j + 1;
					w = j;
				}
				// Determine Givens coefficients for eliminating systemMatrix(i,j) with Pivot element systemMatrix(v,vv)
				rho = std::pow(systemMatrix(j + off, vv), 2) + std::pow(systemMatrix(i, j), 2);
				rho = std::sqrt(rho);
				if (systemMatrix(j + off, vv) < 0)
				{
					rho = -rho;
				}
				// store the coefficients
				C(q, j) = systemMatrix(j + off, vv) / rho;
				S(q, j) = systemMatrix(i, j) / rho;
				// update the system matrix accordingly, i.e. apply the Givens rotation to the corresponding matrix rows
				Eigen::MatrixXd upperMatRow = systemMatrix.block(j + off, 0, 1, tt);
				Eigen::MatrixXd lowerMatRow = systemMatrix.block(i, w, 1, tt);
				systemMatrix.block(j + off, 0, 1, tt) = C(q, j) * upperMatRow + S(q, j) * lowerMatRow;
				systemMatrix.block(i, w, 1, tt) = -S(q, j) * upperMatRow + C(q, j) * lowerMatRow;
			}
			if (!isPiecewisePolynomial)
			{
				off++;
			}
		}
	}
}