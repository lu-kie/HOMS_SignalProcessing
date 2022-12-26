#include <gtest/gtest.h>
#include "HigherOrderMS_1D.h"
#include "Interval.h"

namespace HOMS
{
	TEST(HOMS, systemMatrixHigherOrderPotts)
	{
		const int dataLength = 10;

		EXPECT_THROW(computeSystemMatrix(dataLength, 0, std::numeric_limits<double>::infinity()), std::invalid_argument);

		for (int polynomialOrder = 1; polynomialOrder < 10; polynomialOrder++)
		{
			const auto sysMatrix = computeSystemMatrix(dataLength, polynomialOrder, std::numeric_limits<double>::infinity());
			// check size
			EXPECT_EQ(sysMatrix.rows(), dataLength);
			EXPECT_EQ(sysMatrix.cols(), polynomialOrder);

			// check entries
			for (int row = 0; row < sysMatrix.rows(); row++)
			{
				for (int j = polynomialOrder - 1; j >= 0; j--)
				{
					const auto matEntry = sysMatrix(row, j);
					const auto expectedEntry = pow(row + 1, polynomialOrder - j - 1);
					EXPECT_DOUBLE_EQ(matEntry, expectedEntry);
				}
			}
		}
	}

	TEST(HOMS, systemMatrixHigherOrderMS)
	{
		const int dataLength = 10;
		const double smoothnessPenalty = 2;
		const int smoothnessOrder = 3;

		const auto sysMatrix = computeSystemMatrix(dataLength, smoothnessOrder, smoothnessPenalty);

		// check size
		EXPECT_EQ(sysMatrix.rows(), 2 * dataLength - smoothnessOrder);
		EXPECT_EQ(sysMatrix.cols(), smoothnessOrder + 1);

		// check entries
		for (int row = 0; row < dataLength; row++)
		{
			EXPECT_DOUBLE_EQ(sysMatrix(row, 0), 1);
			EXPECT_DOUBLE_EQ(sysMatrix(row, 1), 0);
			EXPECT_DOUBLE_EQ(sysMatrix(row, 2), 0);
			EXPECT_DOUBLE_EQ(sysMatrix(row, 3), 0);
		}

		for (int row = dataLength; row < sysMatrix.rows(); row++)
		{
			EXPECT_DOUBLE_EQ(sysMatrix(row, 0), -1 * pow(smoothnessPenalty, smoothnessOrder));
			EXPECT_DOUBLE_EQ(sysMatrix(row, 1), 3 * pow(smoothnessPenalty, smoothnessOrder));
			EXPECT_DOUBLE_EQ(sysMatrix(row, 2), -3 * pow(smoothnessPenalty, smoothnessOrder));
			EXPECT_DOUBLE_EQ(sysMatrix(row, 3), 1 * pow(smoothnessPenalty, smoothnessOrder));
		}
	}

	TEST(HOMS, computeGivensCoefficientsPotts)
	{
		const int dataLength = 4;
		const int polynomialOrder = 3;
		Eigen::MatrixXd systemMatrix(dataLength, polynomialOrder);
		systemMatrix << 1, 1, 1,
			4, 2, 1,
			9, 3, 1,
			16, 4, 1;

		const auto givensCoeffs = GivensCoefficients(dataLength, polynomialOrder, std::numeric_limits<double>::infinity());

		// use the computed Givens coefficients to obtain a QR decomposition of systemMatrix
		Eigen::MatrixXd R = systemMatrix;
		Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(dataLength, dataLength);
		for (int row = 0; row < systemMatrix.rows(); row++)
		{
			for (int col = 0; col < row; col++)
			{
				const auto c = givensCoeffs.C(row, col);
				const auto s = givensCoeffs.S(row, col);
				// eliminate matrix entry (row,col)
				const Eigen::VectorXd pivotRow = R.row(col);
				const Eigen::VectorXd targetRow = R.row(row);
				R.row(col) = c * pivotRow + s * targetRow;
				R.row(row) = -s * pivotRow + c * targetRow;
				Eigen::MatrixXd G = Eigen::MatrixXd::Identity(dataLength, dataLength);
				G(col, col) = c;
				G(col, row) = s;
				G(row, col) = -s;
				G(row, row) = c;
				Q = G * Q;
			}
		}

		// check if Q,R yields a QR decomposition of system Matrix
		const Eigen::MatrixXd I = Q.transpose() * Q;
		EXPECT_TRUE(I.isIdentity(1e-12));
		EXPECT_TRUE(R.isUpperTriangular(1e-12));
		EXPECT_TRUE(systemMatrix.isApprox(Q.transpose() * R));
	}

	TEST(HOMS, computeGivensCoefficientsHigherOrderMS)
	{
		const int dataLength = 5;
		const int smoothnessOrder = 2;
		const double smoothnessPenalty = 2;
		Eigen::MatrixXd fullSystemMatrix(2 * dataLength - smoothnessOrder, dataLength);
		fullSystemMatrix <<
			1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1,
			4, -8, 4, 0, 0,
			0, 4, -8, 4, 0,
			0, 0, 4, -8, 4;

		const auto givensCoeffs = GivensCoefficients(dataLength, smoothnessOrder, smoothnessPenalty);

		// use the computed Givens coefficients to obtain a QR decomposition of fullSystemMatrix
		// we need to incorporate offsets as the Givens coefficients are arranged w.r.t. the sparse representation of the system matrix
		Eigen::MatrixXd R = fullSystemMatrix;
		Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(fullSystemMatrix.rows(), fullSystemMatrix.rows());
		int offset = 0;
		for (int row = dataLength; row < fullSystemMatrix.rows(); row++)
		{
			for (int col = offset; col < offset + smoothnessOrder + 1; col++)
			{
				const auto c = givensCoeffs.C(row - smoothnessOrder - 1, col - offset);
				const auto s = givensCoeffs.S(row - smoothnessOrder - 1, col - offset);

				const Eigen::VectorXd pivotRow = R.row(col);
				const Eigen::VectorXd targetRow = R.row(row);
				R.row(col) = c * pivotRow + s * targetRow;
				R.row(row) = -s * pivotRow + c * targetRow;

				Eigen::MatrixXd G = Eigen::MatrixXd::Identity(fullSystemMatrix.rows(), fullSystemMatrix.rows());
				G(col, col) = c;
				G(col, row) = s;
				G(row, col) = -s;
				G(row, row) = c;
				Q = G * Q;
			}
			offset++;
		}

		// check if Q,R yields a QR decomposition of the full system Matrix
		const Eigen::MatrixXd I = Q.transpose() * Q;
		EXPECT_TRUE(I.isIdentity(1e-12));
		EXPECT_TRUE(R.isUpperTriangular(1e-12));
		EXPECT_TRUE(fullSystemMatrix.isApprox(Q.transpose() * R));
	}

	TEST(HOMS, intervalApproxErrorHigherOrderPotts)
	{
		// fit quadratic polynomial to parabolic data: expect zero approximation error
		const int leftBound = 2;
		double dataPoint = 0.0;
		int polynomialOrder = 3;
		const auto smoothnessPenalty = std::numeric_limits<double>::infinity();

		auto quadraticRegressionInterval = Interval(leftBound, dataPoint, polynomialOrder, smoothnessPenalty);
		EXPECT_EQ(quadraticRegressionInterval.getLength(), 1);

		const auto quadraticGivensCoeffs = GivensCoefficients(6, polynomialOrder, smoothnessPenalty);
		for (const double newDataPoint : {1, 4, 9, 16, 25})
		{
			quadraticRegressionInterval.addNewDataPoint(quadraticGivensCoeffs, newDataPoint);
			EXPECT_NEAR(quadraticRegressionInterval.approxError, 0, 1e-12);
		}
		EXPECT_EQ(quadraticRegressionInterval.getLength(), 6);

		// fit linear polynomial to parabolic data: linear regression f(x) = 5x-13.33
		polynomialOrder = 2;
		auto linearRegressionInterval = Interval(leftBound, dataPoint, polynomialOrder, smoothnessPenalty);
		const auto linearGivensCoeffs = GivensCoefficients(6, polynomialOrder, smoothnessPenalty);
		for (const double newDataPoint : {1, 4, 9, 16, 25})
		{
			linearRegressionInterval.addNewDataPoint(linearGivensCoeffs, newDataPoint);
		}
		const double expectedApproxError = std::pow(3.33, 2) + std::pow(1 - 1.67, 2) + std::pow(4 - 6.67, 2)
			+ std::pow(9 - 11.67, 2) + std::pow(16 - 16.67, 2) + std::pow(25 - 21.67, 2);
		EXPECT_NEAR(linearRegressionInterval.approxError, expectedApproxError, 1e-4);
		EXPECT_EQ(quadraticRegressionInterval.getLength(), 6);
	}

	TEST(HOMS, intervalApproxErrorHigherOrderMS)
	{
		// fit third order discrete spline to parabolic data: expect zero approximation error
		const int leftBound = 2;
		const double dataPoint = 0.0;
		const int smoothnessOrder = 3;
		const auto smoothnessPenalty = 3;

		auto thirdOrderSplineInterval = Interval(leftBound, dataPoint, smoothnessOrder, smoothnessPenalty);
		EXPECT_EQ(thirdOrderSplineInterval.getLength(), 1);

		const auto thirdOrderSplineGivensCoeffs = GivensCoefficients(6, smoothnessOrder, smoothnessPenalty);
		for (const double newDataPoint : {1, 4, 9, 16, 25})
		{
			thirdOrderSplineInterval.addNewDataPoint(thirdOrderSplineGivensCoeffs, newDataPoint);
			EXPECT_NEAR(thirdOrderSplineInterval.approxError, 0, 1e-12);
		}
		EXPECT_EQ(thirdOrderSplineInterval.getLength(), 6);
	}

	TEST(HOMS, intervalApplyGivensRotationToDataHigherOrderPotts)
	{
		// fit quadratic polynomial to parabolic data: expect perfect fit
		const int dataLength = 6;
		Eigen::VectorXd intervalData = Eigen::VectorXd::Zero(dataLength);
		intervalData << 0, 1, 4, 9, 16, 25;
		int polynomialOrder = 3;
		const auto smoothnessPenalty = std::numeric_limits<double>::infinity();
		const int leftBound = 2;
		const int rightBound = 7;

		auto quadraticRegressionInterval = Interval(leftBound, rightBound, polynomialOrder, smoothnessPenalty, intervalData);
		const auto quadraticGivensCoeffs = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);
		
		for (int row = 0; row < dataLength; row++)
		{
			for (int col = 0; col < polynomialOrder; col++)
			{
				quadraticRegressionInterval.applyGivensRotationToData(quadraticGivensCoeffs, row, col);
			}
		}

		intervalData.isApprox(quadraticRegressionInterval.data, 1e-12);

		// fit linear polynomial to parabolic data: linear regression f(x) = 5x-13.33
		polynomialOrder = 2;
		auto linearRegressionInterval = Interval(leftBound, rightBound, polynomialOrder, smoothnessPenalty, intervalData);
		const auto linearGivensCoeffs = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);

		for (int row = 0; row < dataLength; row++)
		{
			for (int col = 0; col < polynomialOrder; col++)
			{
				linearRegressionInterval.applyGivensRotationToData(linearGivensCoeffs, row, col);
			}
		}

		Eigen::VectorXd expectedResultVector = Eigen::VectorXd(dataLength);
		expectedResultVector << -3.33, 1.67, 6.67, 11.67, 16.67, 21.67;
		expectedResultVector.isApprox(linearRegressionInterval.data, 1e-12);
	}

}