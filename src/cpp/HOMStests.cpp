#include <gtest/gtest.h>
#include "Interval.h"
#include "HigherOrderMS_1D.h"
#include "PcwPolynomialPartitioning.h"
#include "PcwSmoothPartitioning.h"

namespace HOMS
{
	TEST(HOMS, intervalApproxErrorPcwPolynomial)
	{
		// fit quadratic polynomial to parabolic data: expect zero approximation error
		const int leftBound = 2;
		double dataPoint = 0.0;
		int polynomialOrder = 3;
		const auto smoothnessPenalty = std::numeric_limits<double>::infinity();

		auto quadraticRegressionInterval = Interval(leftBound, dataPoint, polynomialOrder, smoothnessPenalty);
		EXPECT_EQ(quadraticRegressionInterval.size(), 1);

		const auto quadraticGivensCoeffs = GivensCoefficients(6, polynomialOrder, smoothnessPenalty);
		for (const double newDataPoint : {1, 4, 9, 16, 25})
		{
			quadraticRegressionInterval.addNewDataPoint(quadraticGivensCoeffs, newDataPoint);
			EXPECT_NEAR(quadraticRegressionInterval.approxError, 0, 1e-12);
		}
		EXPECT_EQ(quadraticRegressionInterval.size(), 6);

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
		EXPECT_EQ(quadraticRegressionInterval.size(), 6);
	}

	TEST(HOMS, intervalApproxErrorHigherOrderMS)
	{
		// fit third order discrete spline to parabolic data: expect zero approximation error
		const int leftBound = 2;
		const double dataPoint = 0.0;
		const int smoothingOrder = 3;
		const auto smoothnessPenalty = 3;

		auto thirdOrderSplineInterval = Interval(leftBound, dataPoint, smoothingOrder, smoothnessPenalty);
		EXPECT_EQ(thirdOrderSplineInterval.size(), 1);

		const auto thirdOrderSplineGivensCoeffs = GivensCoefficients(6, smoothingOrder, smoothnessPenalty);
		for (const double newDataPoint : {1, 4, 9, 16, 25})
		{
			thirdOrderSplineInterval.addNewDataPoint(thirdOrderSplineGivensCoeffs, newDataPoint);
			EXPECT_NEAR(thirdOrderSplineInterval.approxError, 0, 1e-12);
		}
		EXPECT_EQ(thirdOrderSplineInterval.size(), 6);
	}

	TEST(HOMS, intervalApplyGivensRotationToDataPcwPolynomial)
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

		Eigen::MatrixXd Rquadratic = computeSystemMatrix(dataLength, polynomialOrder, smoothnessPenalty);
		for (int row = 0; row < dataLength; row++)
		{
			for (int col = 0; col < polynomialOrder; col++)
			{
				quadraticRegressionInterval.applyGivensRotationToData(quadraticGivensCoeffs, row, col);
				if (col >= row)
				{
					continue;
				}
				const auto c = quadraticGivensCoeffs.C(row, col);
				const auto s = quadraticGivensCoeffs.S(row, col);
				// eliminate matrix entry (row,col)
				const Eigen::VectorXd pivotRow = Rquadratic.row(col);
				const Eigen::VectorXd targetRow = Rquadratic.row(row);
				Rquadratic.row(col) = c * pivotRow + s * targetRow;
				Rquadratic.row(row) = -s * pivotRow + c * targetRow;
			}
		}
		// obtain the polynomial coefficients and check the corresponding fitted values
		Eigen::MatrixXd RquadraticUpper = Rquadratic.triangularView<Eigen::Upper>();
		Eigen::VectorXd quadraticPolyCoeff = RquadraticUpper.colPivHouseholderQr().solve(quadraticRegressionInterval.data);
		for (int idx = 0; idx < dataLength; idx++)
		{
			const auto x = static_cast<double>(idx + 1);
			const auto computedResult = quadraticPolyCoeff(0) * pow(x, 2) + quadraticPolyCoeff(1) * x + quadraticPolyCoeff(2);
			EXPECT_NEAR(computedResult, intervalData(idx), 1e-12);
		}

		// fit linear polynomial to parabolic data: linear regression f(x) = 5x-8.333 when x=1,...,6
		polynomialOrder = 2;
		auto linearRegressionInterval = Interval(leftBound, rightBound, polynomialOrder, smoothnessPenalty, intervalData);
		const auto linearGivensCoeffs = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);
		Eigen::MatrixXd Rlinear = computeSystemMatrix(dataLength, polynomialOrder, smoothnessPenalty);
		for (int row = 0; row < dataLength; row++)
		{
			for (int col = 0; col < polynomialOrder; col++)
			{
				linearRegressionInterval.applyGivensRotationToData(linearGivensCoeffs, row, col);
				if (col >= row)
				{
					continue;
				}
				const auto c = linearGivensCoeffs.C(row, col);
				const auto s = linearGivensCoeffs.S(row, col);
				// eliminate matrix entry (row,col)
				const Eigen::VectorXd pivotRow = Rlinear.row(col);
				const Eigen::VectorXd targetRow = Rlinear.row(row);
				Rlinear.row(col) = c * pivotRow + s * targetRow;
				Rlinear.row(row) = -s * pivotRow + c * targetRow;
			}
		}

		// obtain linear coefficients and check them
		Eigen::MatrixXd RlinearUpper = Rlinear.triangularView<Eigen::Upper>();
		Eigen::VectorXd linearPolyCoeff = RlinearUpper.colPivHouseholderQr().solve(linearRegressionInterval.data);

		EXPECT_NEAR(linearPolyCoeff(0), 5, 1e-3);
		EXPECT_NEAR(linearPolyCoeff(1), -8.333, 1e-3);
	}

	TEST(HOMS, intervalApplyGivensRotationToDataHigherMS)
	{
		// fit third order discrete spline to parabolic data: expect perfect fit
		const int dataLength = 7;
		Eigen::VectorXd intervalData = Eigen::VectorXd::Zero(dataLength);
		intervalData << 0, 1, 4, 9, 16, 25, 36;
		int smoothingOrder = 3;
		double smoothnessPenalty = 1;

		const int leftBound = 2;
		const int rightBound = 8;

		auto thirdOrderDiscreteSplineInterval = Interval(leftBound, rightBound, smoothingOrder, smoothnessPenalty, intervalData);
		const auto thirdOrderDiscreteSplineGivensCoeffs = GivensCoefficients(dataLength, smoothingOrder, smoothnessPenalty);

		Eigen::MatrixXd fullSystemMatrix(2 * dataLength - smoothingOrder, dataLength);
		fullSystemMatrix <<
			1, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 1,
			-1, 3, -3, 1, 0, 0, 0,
			0, -1, 3, -3, 1, 0, 0,
			0, 0, -1, 3, -3, 1, 0,
			0, 0, 0, -1, 3, -3, 1;

		int colOffset = 0;
		const int rowOffset = dataLength - smoothingOrder;
		for (int row = dataLength; row < fullSystemMatrix.rows(); row++)
		{
			for (int col = colOffset; col < colOffset + smoothingOrder + 1; col++)
			{
				thirdOrderDiscreteSplineInterval.applyGivensRotationToData(thirdOrderDiscreteSplineGivensCoeffs, row, col - colOffset, rowOffset, colOffset);

				const auto c = thirdOrderDiscreteSplineGivensCoeffs.C(row - rowOffset, col - colOffset);
				const auto s = thirdOrderDiscreteSplineGivensCoeffs.S(row - rowOffset, col - colOffset);
				// eliminate matrix entry (row,col)
				const Eigen::VectorXd pivotRow = fullSystemMatrix.row(col);
				const Eigen::VectorXd targetRow = fullSystemMatrix.row(row);
				fullSystemMatrix.row(col) = c * pivotRow + s * targetRow;
				fullSystemMatrix.row(row) = -s * pivotRow + c * targetRow;
			}
			colOffset++;
		}
		Eigen::MatrixXd RUpper = fullSystemMatrix.triangularView<Eigen::Upper>();
		Eigen::VectorXd smoothedData = RUpper.colPivHouseholderQr().solve(thirdOrderDiscreteSplineInterval.data);

		EXPECT_TRUE(intervalData.isApprox(smoothedData, 1e-12));
	}

	TEST(PcwPolynomialPartitioning, computeSystemMatrix)
	{
		const int dataLength = 10;

		for (int polynomialOrder = 1; polynomialOrder < 10; polynomialOrder++)
		{
			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength);
			const auto sysMatrix = pcwImpl.computeSystemMatrix();
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

	TEST(PcwPolynomialPartitioning, createGivensCoefficients)
	{
		const int dataLength = 4;
		const int polynomialOrder = 3;
		Eigen::MatrixXd systemMatrix(dataLength, polynomialOrder);
		systemMatrix << 1, 1, 1,
			4, 2, 1,
			9, 3, 1,
			16, 4, 1;

		auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength);
		const auto givensCoeffs = pcwImpl.createGivensCoefficients();

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

	TEST(PcwPolynomialPartitioning, computeOptimalEnergiesNoSegmentation)
	{
		const auto smoothnessPenalty = std::numeric_limits<double>::infinity();
		const int dataLength = 6;
		Eigen::VectorXd data = Eigen::VectorXd::Zero(dataLength);
		data << 0, 1, 4, 9, 16, 25;

		// quadratic regression for parabolic data -> expect only zero optimal energies
		int polynomialOrder = 3;
		const auto givensCoeffsQuadratic = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);
		auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, givensCoeffsQuadratic);
		const auto quadraticApproximationErrors = pcwImpl.computeOptimalEnergiesNoSegmentation(data);
		for (const auto& err : quadraticApproximationErrors)
		{
			EXPECT_NEAR(err, 0, 1e-12);
		}

		// linear regression for parabolic data
		polynomialOrder = 2;
		const auto givensCoeffsLinear = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);
		pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, givensCoeffsLinear);
		const auto linearApproximationErrors = pcwImpl.computeOptimalEnergiesNoSegmentation(data);

		// aux function for verifying the results
		auto computeExpectedLinearApproxErr = [&data](int idx, double optimalSlope, double optimalOffset)
		{
			double approxErr = 0;
			for (int i = 0; i <= idx; i++)
			{
				approxErr += std::pow((optimalOffset + double(i + 1) * optimalSlope) - data[i], 2);
			}
			return approxErr;
		};

		EXPECT_NEAR(linearApproximationErrors[0], 0, 1e-12);
		EXPECT_NEAR(linearApproximationErrors[1], 0, 1e-12);

		double optimalSlope = 2;
		double optimalOffset = -2.333;
		double expectedApproxErr = computeExpectedLinearApproxErr(2, optimalSlope, optimalOffset);

		EXPECT_NEAR(linearApproximationErrors[2], expectedApproxErr, 1e-3);

		optimalSlope = 3;
		optimalOffset = -4;
		expectedApproxErr = computeExpectedLinearApproxErr(3, optimalSlope, optimalOffset);

		EXPECT_NEAR(linearApproximationErrors[3], expectedApproxErr, 1e-3);

		optimalSlope = 4;
		optimalOffset = -6;
		expectedApproxErr = computeExpectedLinearApproxErr(4, optimalSlope, optimalOffset);

		EXPECT_NEAR(linearApproximationErrors[4], expectedApproxErr, 1e-3);

		optimalSlope = 5;
		optimalOffset = -8.333;
		expectedApproxErr = computeExpectedLinearApproxErr(5, optimalSlope, optimalOffset);

		EXPECT_NEAR(linearApproximationErrors[5], expectedApproxErr, 1e-3);
	}

	TEST(PcwPolynomialPartitioning, findOptimalPartition)
	{
		// piecewise constant
		int dataLength = 10;
		int polynomialOrder = 1;
		auto smoothnessPenalty = std::numeric_limits<double>::infinity();
		auto givensCoeffs = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);

		Eigen::VectorXd data = Eigen::VectorXd::Zero(dataLength);
		data << 1, 1, 1, 1, 1, 10, 10, 10, 10, 10;

		for (const double& jumpPenalty : { 1.0,10.0,50.0,202.0,203.0,250.0 })
		{
			const auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, givensCoeffs);
			const auto foundPartition = pcwImpl.findOptimalPartition(data);
			if (jumpPenalty < 202.5)
			{
				// two segments are optimal
				EXPECT_EQ(foundPartition.size(), 2);
				EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 4));
				EXPECT_EQ(foundPartition.segments.at(1), Segment(5, 9));
			}
			else
			{
				// single segment is optimal
				EXPECT_EQ(foundPartition.size(), 1);
				EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 9));
			}
		}

		// single element segments are optimal for zero jumpPenalty
		data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
		double jumpPenalty = 0;
		auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, givensCoeffs);
		auto foundPartition = pcwImpl.findOptimalPartition(data);

		EXPECT_EQ(foundPartition.size(), dataLength);
		for (int i = 0; i < dataLength; i++)
		{
			EXPECT_EQ(foundPartition.segments.at(i), Segment(i, i));
		}

		// only one segment for constant data 
		dataLength = 1000;
		Eigen::VectorXd constantData = 80 * Eigen::VectorXd::Ones(dataLength);
		givensCoeffs = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);
		for (const double& jumpPenalty : { 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 })
		{
			pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, givensCoeffs);
			foundPartition = pcwImpl.findOptimalPartition(constantData);
			EXPECT_EQ(foundPartition.size(), 1);
			EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 999));
		}

		// piecewise quadratic
		dataLength = 10;
		polynomialOrder = 3;
		smoothnessPenalty = std::numeric_limits<double>::infinity();
		givensCoeffs = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);

		data = Eigen::VectorXd::Zero(dataLength);
		data << 0, 1, 4, 9, 16, -4, -9, -16, -25, -36;
		const auto noJumpEnergy = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, givensCoeffs).computeOptimalEnergiesNoSegmentation(data)[dataLength - 1];

		for (const double& jumpPenalty : { 1.0,10.0,50.0,202.0,203.0,250.0 })
		{
			pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, givensCoeffs);
			const auto foundPartition = pcwImpl.findOptimalPartition(data);

			if (jumpPenalty < noJumpEnergy)
			{
				// two segments are optimal
				EXPECT_EQ(foundPartition.size(), 2);
				EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 4));
				EXPECT_EQ(foundPartition.segments.at(1), Segment(5, 9));
			}
			else
			{
				// single segment is optimal
				EXPECT_EQ(foundPartition.size(), 1);
				EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 9));
			}
		}

		// segments of size three are optimal for near-zero jumpPenalty and pcw. quadratic
		data << 1, -1, 1, -1, 1, -1, 1, -1, 1, -1;
		jumpPenalty = 1e-8;
		pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, givensCoeffs);
		foundPartition = pcwImpl.findOptimalPartition(data);

		EXPECT_EQ(foundPartition.size(), 4);

		int sumSegmentLengths = 0;
		for (const auto& segment : foundPartition.segments)
		{
			const auto segmentLength = segment.size();
			EXPECT_TRUE(segmentLength == 1 || segmentLength == 3);
			sumSegmentLengths += segmentLength;
		}
		EXPECT_EQ(sumSegmentLengths, dataLength);

		// only one segment for quadratic data
		dataLength = 1000;
		constantData = 150 * Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength).cwiseProduct(Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength));
		givensCoeffs = GivensCoefficients(dataLength, polynomialOrder, smoothnessPenalty);
		for (const double& jumpPenalty : { 0.001,0.01,0.1, 1.0,10.0,100.0,1000.0 })
		{
			pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, givensCoeffs);
			foundPartition = pcwImpl.findOptimalPartition(constantData);
			EXPECT_EQ(foundPartition.segments.size(), 1);
			EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 999));
		}

	}

	TEST(PcwSmoothPartitioning, computeSystemMatrix)
	{
		const int dataLength = 10;
		const double smoothnessPenalty = 2;
		const int smoothingOrder = 3;
		auto pcwPartImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength);
		const auto sysMatrix = pcwPartImpl.computeSystemMatrix();

		// check size
		EXPECT_EQ(sysMatrix.rows(), 2 * dataLength - smoothingOrder);
		EXPECT_EQ(sysMatrix.cols(), smoothingOrder + 1);

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
			EXPECT_DOUBLE_EQ(sysMatrix(row, 0), -1 * pow(smoothnessPenalty, smoothingOrder));
			EXPECT_DOUBLE_EQ(sysMatrix(row, 1), 3 * pow(smoothnessPenalty, smoothingOrder));
			EXPECT_DOUBLE_EQ(sysMatrix(row, 2), -3 * pow(smoothnessPenalty, smoothingOrder));
			EXPECT_DOUBLE_EQ(sysMatrix(row, 3), 1 * pow(smoothnessPenalty, smoothingOrder));
		}
	}

	TEST(PcwSmoothPartitioning, createGivensCoefficients)
	{
		const int dataLength = 5;
		const int smoothingOrder = 2;
		const double smoothnessPenalty = 2;
		Eigen::MatrixXd fullSystemMatrix(2 * dataLength - smoothingOrder, dataLength);
		fullSystemMatrix <<
			1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1,
			4, -8, 4, 0, 0,
			0, 4, -8, 4, 0,
			0, 0, 4, -8, 4;

		auto pcwPartImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength);
		const auto givensCoeffs = pcwPartImpl.createGivensCoefficients();

		// use the computed Givens coefficients to obtain a QR decomposition of fullSystemMatrix
		// we need to incorporate offsets as the Givens coefficients are arranged w.r.t. the sparse representation of the system matrix
		Eigen::MatrixXd R = fullSystemMatrix;
		Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(fullSystemMatrix.rows(), fullSystemMatrix.rows());
		int offset = 0;
		for (int row = dataLength; row < fullSystemMatrix.rows(); row++)
		{
			for (int col = offset; col < offset + smoothingOrder + 1; col++)
			{
				const auto c = givensCoeffs.C(row - smoothingOrder - 1, col - offset);
				const auto s = givensCoeffs.S(row - smoothingOrder - 1, col - offset);

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

	TEST(PcwSmoothPartitioning, computeOptimalEnergiesNoSegmentation)
	{
		const int smoothingOrder = 3;
		const auto smoothnessPenalty = 4;
		const int dataLength = 6;
		const auto givensCoeffs = GivensCoefficients(dataLength, smoothingOrder, smoothnessPenalty);
		auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength, givensCoeffs);

		Eigen::VectorXd data = Eigen::VectorXd::Zero(dataLength);
		data << 0, 1, 4, 9, 16, 25;

		// discrete third order smoothing spline for parabolic data -> expect only zero optimal energies
		const auto approximationErrors = pcwImpl.computeOptimalEnergiesNoSegmentation(data);
		for (const auto& err : approximationErrors)
		{
			EXPECT_NEAR(err, 0, 1e-12);
		}
	}

	TEST(PcwSmoothPartitioning, findOptimalPartition)
	{
		int dataLength = 10;
		const int smoothingOrder = 3;
		auto smoothnessPenalty = 20;
		auto givensCoeffs = GivensCoefficients(dataLength, smoothingOrder, smoothnessPenalty);

		Eigen::VectorXd data = Eigen::VectorXd::Zero(dataLength);
		data << 0, 1, 4, 9, 16, -4, -9, -16, -25, -36;
		const auto noJumpEnergy = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength, givensCoeffs).computeOptimalEnergiesNoSegmentation(data)[dataLength - 1];

		for (const double& jumpPenalty : { 1.0,10.0,50.0,202.0,203.0,250.0 })
		{
			auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, jumpPenalty, dataLength, givensCoeffs);
			const auto foundPartition = pcwImpl.findOptimalPartition(data);

			if (jumpPenalty < noJumpEnergy)
			{
				// two segments are optimal
				EXPECT_EQ(foundPartition.size(), 2);
				EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 4));
				EXPECT_EQ(foundPartition.segments.at(1), Segment(5, 9));
			}
			else
			{
				// single segment is optimal
				EXPECT_EQ(foundPartition.size(), 1);
				EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 9));
			}
		}

		// segments of size three are optimal for near-zero jumpPenalty and pcw. quadratic
		data << 1, -1, 1, -1, 1, -1, 1, -1, 1, -1;
		double jumpPenalty = 1e-8;
		auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, jumpPenalty, dataLength, givensCoeffs);
		auto foundPartition = pcwImpl.findOptimalPartition(data);

		EXPECT_EQ(foundPartition.size(), 4);

		int sumSegmentLengths = 0;
		for (const auto& segment : foundPartition.segments)
		{
			const auto segmentLength = segment.size();
			EXPECT_TRUE(segmentLength == 1 || segmentLength == 3);
			sumSegmentLengths += segmentLength;
		}
		EXPECT_EQ(sumSegmentLengths, dataLength);

		// only one segment for quadratic data
		dataLength = 500;
		Eigen::VectorXd quadraticData = 150 * Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength).cwiseProduct(Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength));

		for (const double& jumpPenalty : { 0.1,1.0,10.0,100.0,1000.0 })
		{
			for (const double& smoothnessPenalty : { 0.01,0.1,1.0,10.0 })
			{
				givensCoeffs = GivensCoefficients(dataLength, smoothingOrder, smoothnessPenalty);
				pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, jumpPenalty, dataLength, givensCoeffs);
				auto errs = pcwImpl.computeOptimalEnergiesNoSegmentation(quadraticData);
				auto foundPartition = pcwImpl.findOptimalPartition(quadraticData);
				EXPECT_EQ(foundPartition.size(), 1);
				EXPECT_EQ(foundPartition.segments.at(0), Segment(0, dataLength - 1));
			}
		}
	}

	TEST(PcwPolynomialPartitioning, pcwPolynomialResult)
	{
		{
			// piecewise quadratic, perfect fit
			int dataLength = 15;
			Eigen::VectorXd data = Eigen::VectorXd::Zero(dataLength);
			data << 0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50;

			Partitioning expectedPartition;
			expectedPartition.segments.push_back(Segment(0, 5));
			expectedPartition.segments.push_back(Segment(6, 12));
			expectedPartition.segments.push_back(Segment(13, 14));

			const int polynomialOrder = 3;
			const double jumpPenalty = 0.1;
			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength);
			auto [pcwPolynomialResult, partition] = pcwImpl.applyToData(data);

			EXPECT_EQ(partition.size(), expectedPartition.size());
			EXPECT_EQ(partition.segments.at(0), expectedPartition.segments.at(0));
			EXPECT_EQ(partition.segments.at(1), expectedPartition.segments.at(1));
			EXPECT_EQ(partition.segments.at(2), expectedPartition.segments.at(2));

			EXPECT_EQ(dataLength, static_cast<int>(pcwPolynomialResult.size()));
			EXPECT_TRUE(data.isApprox(pcwPolynomialResult, 1e-8));
		}

		{
			// piecewise linear, first segment f(x) = 5x-13.33, second segment perfect fit
			int dataLength = 15;
			const int polynomialOrder = 2;
			const double jumpPenalty = 50;

			Eigen::VectorXd data = Eigen::VectorXd::Zero(dataLength);
			
			data << 0, 1, 4, 9, 16, 25, 100, 90, 80, 70, 60, 50, 40, 30, 20;

			Partitioning expectedPartition;
			expectedPartition.segments.push_back(Segment(0, 5));
			expectedPartition.segments.push_back(Segment(6, 14));
			

			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength);
			auto [pcwPolynomialResult, partition] = pcwImpl.applyToData(data);

			EXPECT_EQ(partition.size(), expectedPartition.size());
			EXPECT_EQ(partition.segments.at(0), expectedPartition.segments.at(0));
			EXPECT_EQ(partition.segments.at(1), expectedPartition.segments.at(1));

			EXPECT_EQ(dataLength, static_cast<int>(pcwPolynomialResult.size()));
			Eigen::VectorXd expectedResult = Eigen::VectorXd::Zero(dataLength);
			expectedResult << -3.33, 1.67, 6.67, 11.67, 16.67, 21.67, 100, 90, 80, 70, 60, 50, 40, 30, 20;
			EXPECT_TRUE(expectedResult.isApprox(pcwPolynomialResult, 1e-4));
		}
	}

	TEST(PcwSmoothPartitioning, pcwSmoothResult)
	{
		// piecewise quadratic, perfect fit for third order smoothness
		int dataLength = 15;
		Eigen::VectorXd data = Eigen::VectorXd::Zero(dataLength);
		data << 0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50;

		int smoothingOrder = 3;
		const auto smoothnessPenalty = 1;

		auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 0.1, dataLength);
		auto [result, partition] = pcwImpl.applyToData(data);

		Partitioning expectedPartition;
		expectedPartition.segments.push_back(Segment(0, 5));
		expectedPartition.segments.push_back(Segment(6, 12));
		expectedPartition.segments.push_back(Segment(13, 14));
		Eigen::VectorXd expectedResult = data;

		EXPECT_EQ(dataLength, static_cast<int>(result.size()));
		EXPECT_TRUE(data.isApprox(result, 1e-8));

		EXPECT_EQ(partition.size(), expectedPartition.size());
		EXPECT_EQ(partition.segments.at(0), expectedPartition.segments.at(0));
		EXPECT_EQ(partition.segments.at(1), expectedPartition.segments.at(1));
		EXPECT_EQ(partition.segments.at(2), expectedPartition.segments.at(2));
	}
}