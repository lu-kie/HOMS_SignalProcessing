#include <gtest/gtest.h>
#include "HelperStructs.h"
#include "PcwPolynomialPartitioning.h"
#include "PcwSmoothPartitioning.h"

namespace homs
{

	TEST(ApproxIntervalPolynomial, approxError)
	{
		{
			// fit quadratic polynomial to parabolic data: expect zero approximation error
			const int leftBound = 2;
			const int numChannels = 5;
			const Eigen::VectorXd dataPoint{ {0,36,0,0,36 } };
			const int polynomialOrder = 3;
			const int fullDataLength = 6;

			auto quadraticRegressionInterval = ApproxIntervalPolynomial(leftBound, dataPoint, polynomialOrder, numChannels);
			EXPECT_EQ(quadraticRegressionInterval.size(), 1);

			auto dummyGivensProvider = PcwPolynomialPartitioning(polynomialOrder, 1, fullDataLength, numChannels);
			dummyGivensProvider.initialize();
			const auto quadraticGivensCoeffs = dummyGivensProvider.m_givensCoeffs;

			Eigen::MatrixXd newDataPoints
			{
				{1, 4, 9, 16, 25},
				{25, 16, 9, 4, 1},
				{-1, -4, -9, -16, -25},
				{1, 4, 9, 16, 25},
				{25, 16, 9, 4, 1}
			};
			for (const auto& newDataPoint : newDataPoints.colwise())
			{
				quadraticRegressionInterval.addNewDataPoint(quadraticGivensCoeffs, newDataPoint);
				EXPECT_NEAR(quadraticRegressionInterval.approxError, 0, 1e-12);
			}
			EXPECT_EQ(quadraticRegressionInterval.size(), 6);
		}

		{
			// fit linear polynomial to parabolic data: linear regression f(x) = 5x-13.33
			for (int numChannels = 1; numChannels <= 10; numChannels++)
			{
				const int polynomialOrder = 2;
				const int leftBound = 2;
				const int fullDataLength = 6;
				const Eigen::VectorXd dataPoint = Eigen::VectorXd::Zero(numChannels);
				auto linearRegressionInterval = ApproxIntervalPolynomial(leftBound, dataPoint, polynomialOrder, numChannels);

				auto dummyGivensProvider = PcwPolynomialPartitioning(polynomialOrder, 1, fullDataLength, numChannels);
				dummyGivensProvider.initialize();
				const auto linearGivensCoeffs = dummyGivensProvider.m_givensCoeffs;

				for (const double newDataPointChannelWise : {1, 4, 9, 16, 25})
				{
					linearRegressionInterval.addNewDataPoint(linearGivensCoeffs, newDataPointChannelWise * Eigen::VectorXd::Ones(numChannels));
				}
				double expectedApproxError = std::pow(3.33, 2) + std::pow(1 - 1.67, 2) + std::pow(4 - 6.67, 2)
					+ std::pow(9 - 11.67, 2) + std::pow(16 - 16.67, 2) + std::pow(25 - 21.67, 2);
				expectedApproxError *= numChannels;
				EXPECT_NEAR(linearRegressionInterval.approxError, expectedApproxError, 1e-3);
				EXPECT_EQ(linearRegressionInterval.size(), fullDataLength);
			}
		}
	}

	TEST(ApproxIntervalSmooth, approxError)
	{
		// fit third order discrete spline to parabolic data: expect zero approximation error
		for (int numChannels = 1; numChannels <= 10; numChannels++)
		{
			const int leftBound = 2;
			const Eigen::VectorXd dataPoint = Eigen::VectorXd::Zero(numChannels);
			const int smoothingOrder = 3;
			const auto smoothnessPenalty = 3;
			const int fullDataLength = 6;

			auto thirdOrderSplineInterval = ApproxIntervalSmooth(leftBound, dataPoint, smoothingOrder, smoothnessPenalty, numChannels);

			EXPECT_EQ(thirdOrderSplineInterval.size(), 1);

			auto dummyGivensProvider = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, fullDataLength, numChannels);
			dummyGivensProvider.initialize();
			const auto thirdOrderSplineGivensCoeffs = dummyGivensProvider.m_givensCoeffs;
			for (const double newDataPointChannelWise : {1, 4, 9, 16, 25})
			{
				thirdOrderSplineInterval.addNewDataPoint(thirdOrderSplineGivensCoeffs, newDataPointChannelWise * Eigen::VectorXd::Ones(numChannels));
				EXPECT_NEAR(thirdOrderSplineInterval.approxError, 0, 1e-12);
			}
			EXPECT_EQ(thirdOrderSplineInterval.size(), fullDataLength);
		}
	}

	TEST(ApproxIntervalPolynomial, applyGivensRotationToData)
	{
		{
			// fit quadratic polynomial to parabolic data: expect perfect fit
			const int numChannels = 5;
			const int dataLength = 6;
			Eigen::MatrixXd intervalData
			{
				{1, 4, 9, 16, 25, 36},
				{25, 16, 9, 4, 1, 0},
				{-1, -4, -9, -16, -25, -36},
				{1, 4, 9, 16, 25, 36},
				{25, 16, 9, 4, 1, 0}
			};
			int polynomialOrder = 3;
			const int leftBound = 0;
			const int rightBound = 5;

			auto quadraticRegressionInterval = ApproxIntervalPolynomial(leftBound, rightBound, intervalData, polynomialOrder);

			auto dummyGivensProvider = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
			dummyGivensProvider.initialize();
			const auto quadraticGivensCoeffs = dummyGivensProvider.m_givensCoeffs;

			Eigen::MatrixXd Rquadratic = dummyGivensProvider.createSystemMatrix();
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
			for (int channel = 0; channel < numChannels; channel++)
			{
				Eigen::VectorXd rhs = quadraticRegressionInterval.data.row(channel);
				Eigen::VectorXd quadraticPolyCoeff = RquadraticUpper.colPivHouseholderQr().solve(rhs);
				for (int idx = 0; idx < dataLength; idx++)
				{
					const auto x = static_cast<double>(idx + 1);
					const auto computedResult = quadraticPolyCoeff(0) * pow(x, 2) + quadraticPolyCoeff(1) * x + quadraticPolyCoeff(2);
					EXPECT_NEAR(computedResult, intervalData(channel, idx), 1e-12);
				}
			}
		}

		{
			// fit linear polynomial to parabolic data: linear regression f(x) = 7x-9.333 when x=1,...,6
			const int polynomialOrder = 2;
			const int dataLength = 6;
			const int numChannels = 5;
			const int leftBound = 0;
			const int rightBound = 5;
			Eigen::MatrixXd intervalData
			{
				{1, 4, 9, 16, 25, 36},
				{1, 4, 9, 16, 25, 36},
				{1, 4, 9, 16, 25, 36},
				{1, 4, 9, 16, 25, 36},
				{1, 4, 9, 16, 25, 36}
			};
			auto linearRegressionInterval = ApproxIntervalPolynomial(leftBound, rightBound, intervalData, polynomialOrder);

			auto dummyGivensProvider = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
			dummyGivensProvider.initialize();
			const auto linearGivensCoeffs = dummyGivensProvider.m_givensCoeffs;
			Eigen::MatrixXd Rlinear = dummyGivensProvider.createSystemMatrix();
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
			const Eigen::MatrixXd RlinearUpper = Rlinear.triangularView<Eigen::Upper>();
			for (int channel = 0; channel < numChannels; channel++)
			{
				const Eigen::VectorXd rhs = linearRegressionInterval.data.row(channel);
				const Eigen::VectorXd linearPolyCoeff = RlinearUpper.colPivHouseholderQr().solve(rhs);

				EXPECT_NEAR(linearPolyCoeff(0), 7, 1e-3);
				EXPECT_NEAR(linearPolyCoeff(1), -9.333, 1e-3);
			}
		}
	}

	TEST(ApproxIntervalSmooth, applyGivensRotationToData)
	{
		// fit third order discrete spline to parabolic data: expect perfect fit
		const int dataLength = 7;
		const int numChannels = 5;
		Eigen::MatrixXd intervalData
		{
			{0, 1, 4, 9, 16, 25, 36},
			{0, 1, 4, 9, 16, 25, 36},
			{0, 1, 4, 9, 16, 25, 36},
			{0, 1, 4, 9, 16, 25, 36},
			{0, 1, 4, 9, 16, 25, 36}
		};
		int smoothingOrder = 3;
		double smoothnessPenalty = 1;

		const int leftBound = 0;
		const int rightBound = 6;

		auto thirdOrderDiscreteSplineInterval = ApproxIntervalSmooth(leftBound, rightBound, intervalData, smoothingOrder, smoothnessPenalty);

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

		auto dummyGivensProvider = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength, numChannels);
		dummyGivensProvider.initialize();
		const auto thirdOrderDiscreteSplineGivensCoeffs = dummyGivensProvider.m_givensCoeffs;
		for (int row = 0; row < fullSystemMatrix.rows(); row++)
		{
			for (int col = 0; col < smoothingOrder + 1; col++)
			{
				thirdOrderDiscreteSplineInterval.applyGivensRotationToData(thirdOrderDiscreteSplineGivensCoeffs, row, col);

				if (row >= dataLength)
				{
					const auto c = thirdOrderDiscreteSplineGivensCoeffs.C(row - dataLength + smoothingOrder, col);
					const auto s = thirdOrderDiscreteSplineGivensCoeffs.S(row - dataLength + smoothingOrder, col);
					// eliminate matrix entry (row,col)
					const Eigen::VectorXd pivotRow = fullSystemMatrix.row(col - dataLength + row);
					const Eigen::VectorXd targetRow = fullSystemMatrix.row(row);
					fullSystemMatrix.row(col - dataLength + row) = c * pivotRow + s * targetRow;
					fullSystemMatrix.row(row) = -s * pivotRow + c * targetRow;
				}

			}
			colOffset++;
		}

		const Eigen::MatrixXd RUpper = fullSystemMatrix.triangularView<Eigen::Upper>();
		for (int channel = 0; channel < numChannels; channel++)
		{
			const Eigen::VectorXd rhs = thirdOrderDiscreteSplineInterval.data.row(channel);
			const Eigen::VectorXd smoothedData = RUpper.colPivHouseholderQr().solve(rhs);
			const Eigen::VectorXd expected = intervalData.row(channel);
			EXPECT_TRUE(expected.isApprox(smoothedData, 1e-12));
		}
	}

	TEST(PcwPolynomialPartitioning, createSystemMatrix)
	{
		const int dataLength = 10;
		const int numChannels = 4;
		for (int polynomialOrder = 1; polynomialOrder < 10; polynomialOrder++)
		{
			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
			const auto sysMatrix = pcwImpl.createSystemMatrix();
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

	TEST(PcwPolynomialPartitioning, computeGivensCoefficients)
	{
		const int dataLength = 4;
		const int numChannels = 5;
		const int polynomialOrder = 3;
		Eigen::MatrixXd systemMatrix{
			{1,  1, 1},
			{4,  2, 1},
			{9,  3, 1},
			{16, 4, 1}
		};

		auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
		pcwImpl.initialize();
		const auto givensCoeffs = pcwImpl.m_givensCoeffs;

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
		const int dataLength = 6;
		const int numChannels = 5;
		Eigen::MatrixXd data
		{
			{1, 4, 9, 16, 25, 36},
			{25, 16, 9, 4, 1, 0},
			{-1, -4, -9, -16, -25, -36},
			{1, 4, 9, 16, 25, 36},
			{25, 16, 9, 4, 1, 0}
		};
		{
			// quadratic regression for parabolic data -> expect only zero optimal energies
			int polynomialOrder = 3;
			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
			pcwImpl.initialize();
			const auto quadraticApproximationErrors = pcwImpl.computeOptimalEnergiesNoSegmentation(data);
			for (const auto& err : quadraticApproximationErrors)
			{
				EXPECT_NEAR(err, 0, 1e-12);
			}
		}

		{
			// linear regression for parabolic data
			const int polynomialOrder = 2;
			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
			pcwImpl.initialize();
			const auto linearApproximationErrors = pcwImpl.computeOptimalEnergiesNoSegmentation(data);

			// aux function for verifying the results
			auto computeExpectedLinearApproxErr = [&data](int idx, double optimalSlope, double optimalOffset)
			{
				double approxErr = 0;
				for (int i = 0; i <= idx; i++)
				{

					approxErr += std::pow((optimalOffset + double(i + 1) * optimalSlope) - data(0, i), 2);

				}
				return approxErr;
			};

			EXPECT_NEAR(linearApproximationErrors[0], 0, 1e-12);
			EXPECT_NEAR(linearApproximationErrors[1], 0, 1e-12);

			double optimalSlope = 4;
			double optimalOffset = -3.333;
			double expectedApproxErr = numChannels * computeExpectedLinearApproxErr(2, optimalSlope, optimalOffset);

			EXPECT_NEAR(linearApproximationErrors[2], expectedApproxErr, 1e-3);

			optimalSlope = 5;
			optimalOffset = -5;
			expectedApproxErr = numChannels * computeExpectedLinearApproxErr(3, optimalSlope, optimalOffset);

			EXPECT_NEAR(linearApproximationErrors[3], expectedApproxErr, 1e-3);

			optimalSlope = 6;
			optimalOffset = -7;
			expectedApproxErr = numChannels * computeExpectedLinearApproxErr(4, optimalSlope, optimalOffset);

			EXPECT_NEAR(linearApproximationErrors[4], expectedApproxErr, 1e-3);

			optimalSlope = 7;
			optimalOffset = -9.333;
			expectedApproxErr = numChannels * computeExpectedLinearApproxErr(5, optimalSlope, optimalOffset);

			EXPECT_NEAR(linearApproximationErrors[5], expectedApproxErr, 1e-3);
		}
	}

	TEST(PcwPolynomialPartitioning, findOptimalPartition)
	{
		{
			// piecewise constant
			const int dataLength = 10;
			const int numChannels = 4;
			const int polynomialOrder = 1;

			{
				Eigen::MatrixXd data
				{
					{1, 1, 1, 1, 1, 10, 10, 10, 10, 10},
					{6, 6, 6, 6, 6, 12, 12, 12, 12, 12},
					{0, 0, 0, 0, 0, 8, 8, 8, 8, 8},
					{7, 7, 7, 7, 7, 9, 9, 9, 9, 9}
				};
				auto dummyPcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
				dummyPcwImpl.initialize();
				const auto noJumpEnergy = dummyPcwImpl.computeOptimalEnergiesNoSegmentation(data)[dataLength - 1];
				for (const double& jumpPenalty : { 1.0,10.0,50.0,202.0,noJumpEnergy + 1, 2 * noJumpEnergy })
				{
					auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, numChannels);
					pcwImpl.initialize();
					auto t = pcwImpl.computeOptimalEnergiesNoSegmentation(data);
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
			}
			{
				const int dataLength = 10;
				const int numChannels = 4;
				// single element segments are optimal for zero jumpPenalty
				Eigen::MatrixXd data
				{
					{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
					{ 0, 1, 2, 4, 5, 6, 7, 8, 9, 10 },
					{ 5, 7, 3, 4, 5, 6, 7, 8, 9, 10 },
					{ 9, 8, 3, 4, 5, 6, 7, 8, 9, 10 }
				};
				double jumpPenalty = 0;
				auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, numChannels);
				pcwImpl.initialize();
				auto foundPartition = pcwImpl.findOptimalPartition(data);

				EXPECT_EQ(foundPartition.size(), dataLength);
				for (int i = 0; i < dataLength; i++)
				{
					EXPECT_EQ(foundPartition.segments.at(i), Segment(i, i));
				}
			}
			{
				// only one segment for constant data
				const int dataLength = 1000;
				const int numChannels = 3;
				Eigen::MatrixXd constantData = 80 * Eigen::MatrixXd::Ones(numChannels, dataLength);
				for (const double& jumpPenalty : { 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0 })
				{
					auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, numChannels);
					pcwImpl.initialize();
					const auto foundPartition = pcwImpl.findOptimalPartition(constantData);
					EXPECT_EQ(foundPartition.size(), 1);
					EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 999));
				}
			}
		}
		{
			// piecewise quadratic
			const int dataLength = 10;
			const int numChannels = 4;
			const int polynomialOrder = 3;
			{
				Eigen::MatrixXd data
				{
					{0, 1, 4, 9, 16, -4, -9, -16, -25, -36},
					{0, 1, 4, 9, 16, -4, -9, -16, -25, -36},
					{0, 1, 4, 9, 16, -4, -9, -16, -25, -36},
					{0, 1, 4, 9, 16, -4, -9, -16, -25, -36}
				};
				auto dummyPcwImpl = PcwPolynomialPartitioning(polynomialOrder, 1, dataLength, numChannels);
				dummyPcwImpl.initialize();

				const auto noJumpEnergy = dummyPcwImpl.computeOptimalEnergiesNoSegmentation(data)[dataLength - 1];
				for (const double& jumpPenalty : { 1.0,10.0,50.0,202.0,203.0,250.0 })
				{
					auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, numChannels);
					pcwImpl.initialize();
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
			}

			{
				// segments of size three are optimal for near-zero jumpPenalty and pcw. quadratic
				Eigen::MatrixXd data{ {1, -1, 1, -1, 1, -1, 1, -1, 1, -1} };
				const double jumpPenalty = 1e-8;
				auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, 1);
				pcwImpl.initialize();
				const auto foundPartition = pcwImpl.findOptimalPartition(data);

				EXPECT_EQ(foundPartition.size(), 4);

				int sumSegmentLengths = 0;
				for (const auto& segment : foundPartition.segments)
				{
					const auto segmentLength = segment.size();
					EXPECT_TRUE(segmentLength == 1 || segmentLength == 3);
					sumSegmentLengths += segmentLength;
				}
				EXPECT_EQ(sumSegmentLengths, dataLength);
			}
			{
				// only one segment for quadratic data
				const int dataLength = 1000;
				const int numChannels = 1;
				Eigen::MatrixXd quadraticData = 150 * Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength).cwiseProduct(Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength));
				quadraticData.transposeInPlace();
				for (const double& jumpPenalty : { 0.001,0.01,0.1, 1.0,10.0,100.0,1000.0 })
				{
					auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, numChannels);
					pcwImpl.initialize();
					const auto foundPartition = pcwImpl.findOptimalPartition(quadraticData);
					EXPECT_EQ(foundPartition.segments.size(), 1);
					EXPECT_EQ(foundPartition.segments.at(0), Segment(0, 999));
				}
			}
		}
	}

	TEST(PcwPolynomialPartitioning, pcwPolynomialResult)
	{
		{
			// piecewise quadratic, perfect fit
			const int dataLength = 15;
			const int numChannels = 5;
			Eigen::MatrixXd data
			{
				{0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50},
				{0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50},
				{0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50},
				{0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50},
				{0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50}
			};

			Partitioning expectedPartition;
			expectedPartition.segments.push_back(Segment(0, 5));
			expectedPartition.segments.push_back(Segment(6, 12));
			expectedPartition.segments.push_back(Segment(13, 14));

			const int polynomialOrder = 3;
			const double jumpPenalty = 0.1;
			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, numChannels);
			auto [pcwPolynomialResult, partition] = pcwImpl.applyToData(data);

			EXPECT_EQ(partition.size(), expectedPartition.size());
			EXPECT_EQ(partition.segments.at(0), expectedPartition.segments.at(0));
			EXPECT_EQ(partition.segments.at(1), expectedPartition.segments.at(1));
			EXPECT_EQ(partition.segments.at(2), expectedPartition.segments.at(2));

			EXPECT_EQ(dataLength, static_cast<int>(pcwPolynomialResult.cols()));
			EXPECT_EQ(numChannels, static_cast<int>(pcwPolynomialResult.rows()));
			EXPECT_TRUE(data.isApprox(pcwPolynomialResult, 1e-8));
		}

		{
			// piecewise linear, first segment f(x) = 5x-13.33, second segment perfect fit
			const int dataLength = 15;
			const int numChannels = 3;
			const int polynomialOrder = 2;
			const double jumpPenalty = 150;

			Eigen::MatrixXd data
			{
				{0, 1, 4, 9, 16, 25, 100, 90, 80, 70, 60, 50, 40, 30, 20},
				{0, 1, 4, 9, 16, 25, 100, 90, 80, 70, 60, 50, 40, 30, 20},
				{0, 1, 4, 9, 16, 25, 100, 90, 80, 70, 60, 50, 40, 30, 20}
			};

			Partitioning expectedPartition;
			expectedPartition.segments.push_back(Segment(0, 5));
			expectedPartition.segments.push_back(Segment(6, 14));

			auto pcwImpl = PcwPolynomialPartitioning(polynomialOrder, jumpPenalty, dataLength, numChannels);
			auto [pcwPolynomialResult, partition] = pcwImpl.applyToData(data);

			EXPECT_EQ(partition.size(), expectedPartition.size());
			EXPECT_EQ(partition.segments.at(0), expectedPartition.segments.at(0));
			EXPECT_EQ(partition.segments.at(1), expectedPartition.segments.at(1));

			Eigen::MatrixXd expectedResult
			{
				{ -3.33, 1.67, 6.67, 11.67, 16.67, 21.67, 100, 90, 80, 70, 60, 50, 40, 30, 20 },
				{ -3.33, 1.67, 6.67, 11.67, 16.67, 21.67, 100, 90, 80, 70, 60, 50, 40, 30, 20 },
				{ -3.33, 1.67, 6.67, 11.67, 16.67, 21.67, 100, 90, 80, 70, 60, 50, 40, 30, 20 }
			};

			EXPECT_EQ(dataLength, static_cast<int>(pcwPolynomialResult.cols()));
			EXPECT_EQ(numChannels, static_cast<int>(pcwPolynomialResult.rows()));
			EXPECT_TRUE(expectedResult.isApprox(pcwPolynomialResult, 1e-4));
		}
	}

	TEST(PcwSmoothPartitioning, createSystemMatrix)
	{
		const int dataLength = 10;
		const double smoothnessPenalty = 2;
		const int smoothingOrder = 3;
		auto pcwPartImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength, 1);
		const auto sysMatrix = pcwPartImpl.createSystemMatrix();

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

	TEST(PcwSmoothPartitioning, computeGivensCoefficients)
	{
		const int dataLength = 5;
		const int numChannels = 4;
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

		auto pcwPartImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength, numChannels);
		pcwPartImpl.initialize();
		const auto givensCoeffs = pcwPartImpl.m_givensCoeffs;

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
		const int numChannels = 5;
		auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength, numChannels);
		pcwImpl.initialize();
		Eigen::MatrixXd data
		{
			{0, 1, 4, 9, 16, 25},
			{1, 4, 9, 16, 25, 36},
			{0, 1, 2, 3, 4, 5},
			{5, 5, 5, 5, 5, 5},
			{0, -1, -4, -9, -16, -25},
		};

		// discrete third order smoothing spline for parabolic data -> expect only zero optimal energies
		const auto approximationErrors = pcwImpl.computeOptimalEnergiesNoSegmentation(data);
		for (const auto& err : approximationErrors)
		{
			EXPECT_NEAR(err, 0, 1e-12);
		}
	}

	TEST(PcwSmoothPartitioning, findOptimalPartition)
	{
		{
			const int dataLength = 10;
			const int numChannels = 5;
			const int smoothingOrder = 3;
			const int smoothnessPenalty = 20;

			Eigen::MatrixXd data
			{
				{0, 1, 4, 9, 16, -4, -9, -16, -25, -36},
				{0, 1, 4, 9, 16, -4, -9, -16, -25, -36},
				{0, 1, 4, 9, 16, -4, -9, -16, -25, -36},
				{0, 1, 4, 9, 16, -4, -9, -16, -25, -36},
				{0, 1, 4, 9, 16, -4, -9, -16, -25, -36}
			};
			auto dummyPcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 1, dataLength, numChannels);
			dummyPcwImpl.initialize();
			const auto noJumpEnergy = dummyPcwImpl.computeOptimalEnergiesNoSegmentation(data)[dataLength - 1];

			for (const double& jumpPenalty : { 1.0, 10.0, 50.0,noJumpEnergy - 0.5, noJumpEnergy + 0.5, 2 * noJumpEnergy })
			{
				auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, jumpPenalty, dataLength, numChannels);
				pcwImpl.initialize();
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
		}
		{
			const int dataLength = 10;
			const int numChannels = 5;
			const int smoothingOrder = 3;
			const double smoothnessPenalty = 20;
			// segments of size three are optimal for near-zero jumpPenalty and pcw. quadratic
			Eigen::MatrixXd data
			{
				{1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
				{1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
				{1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
				{1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
				{1, -1, 1, -1, 1, -1, 1, -1, 1, -1}
			};
			double jumpPenalty = 1e-8;
			auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, jumpPenalty, dataLength, numChannels);
			pcwImpl.initialize();
			const auto foundPartition = pcwImpl.findOptimalPartition(data);

			EXPECT_EQ(foundPartition.size(), 4);

			int sumSegmentLengths = 0;
			for (const auto& segment : foundPartition.segments)
			{
				const auto segmentLength = segment.size();
				EXPECT_TRUE(segmentLength == 1 || segmentLength == 3);
				sumSegmentLengths += segmentLength;
			}
			EXPECT_EQ(sumSegmentLengths, dataLength);
			{
				// only one segment for quadratic data
				const int dataLength = 500;
				const int numChannels = 1;
				Eigen::MatrixXd quadraticData = 150 * Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength).cwiseProduct(Eigen::VectorXd::LinSpaced(dataLength, 1, dataLength));
				quadraticData.transposeInPlace();

				for (const double& jumpPenalty : { 0.1,1.0,10.0,100.0,1000.0 })
				{
					for (const double& smoothnessPenalty : { 0.01,0.1,1.0,10.0 })
					{
						pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, jumpPenalty, dataLength, numChannels);
						pcwImpl.initialize();
						const auto foundPartition = pcwImpl.findOptimalPartition(quadraticData);
						EXPECT_EQ(foundPartition.size(), 1);
						EXPECT_EQ(foundPartition.segments.at(0), Segment(0, dataLength - 1));
					}
				}
			}
		}
	}

	TEST(PcwSmoothPartitioning, pcwSmoothResult)
	{
		// piecewise quadratic, perfect fit for third order smoothness
		const int dataLength = 15;
		const int numChannels = 5;
		const int smoothingOrder = 3;
		const auto smoothnessPenalty = 1;
		Eigen::MatrixXd data
		{
			{ 0, 1, 4, 9, 16, 25, -2, -8, -18, -32, -50, -72, -98, 100, 50 },
			{ -2, -8, -18, -32, -50, -72, 0, 1, 4, 9, 16, 25, 36, 50, 100 },
			{ 0, 2, 8, 18, 32, 50, -2, -8, -18, -32, -50, -72, -98, 100, 50 },
			{ 0, 1, 4, 9, 16, 25, -4, -16, -36, -64, -100, -144, -196, 50, 100 },
			{ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196 }
		};

		auto pcwImpl = PcwSmoothPartitioning(smoothingOrder, smoothnessPenalty, 0.1, dataLength, numChannels);
		auto [result, partition] = pcwImpl.applyToData(data);

		Partitioning expectedPartition;
		expectedPartition.segments.push_back(Segment(0, 5));
		expectedPartition.segments.push_back(Segment(6, 12));
		expectedPartition.segments.push_back(Segment(13, 14));
		Eigen::MatrixXd expectedResult = data;

		EXPECT_EQ(dataLength, static_cast<int>(result.cols()));
		EXPECT_EQ(numChannels, static_cast<int>(result.rows()));
		EXPECT_TRUE(data.isApprox(result, 1e-8));

		EXPECT_EQ(partition.size(), expectedPartition.size());
		EXPECT_EQ(partition.segments.at(0), expectedPartition.segments.at(0));
		EXPECT_EQ(partition.segments.at(1), expectedPartition.segments.at(1));
		EXPECT_EQ(partition.segments.at(2), expectedPartition.segments.at(2));
	}
}