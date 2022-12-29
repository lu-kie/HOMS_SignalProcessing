#include "HigherOrderMS_1D.h"
#include "Interval.h"
#include <limits>
#include <iostream>

namespace HOMS
{
	GivensCoefficients::GivensCoefficients(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty)
	{
		const bool isPcwPolynomial = std::isinf(smoothnessPenalty);
		if (isPcwPolynomial)
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
			if (!isPcwPolynomial && i < dataLength)
			{
				continue;
			}

			for (int j = 0; j < smoothnessOrder + 1; j++)
			{
				if (isPcwPolynomial && (j == smoothnessOrder || i <= j))
				{
					break;
				}

				// systemMatrix(v, vv): Pivot element to eliminate the entry systemMatrix(i,j)
				// C(q,j), S(q,j): locations to store the corresponding Givens coefficients
				if (isPcwPolynomial)
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
			if (!isPcwPolynomial)
			{
				off++;
			}
		}
	}

	Partitioning::Partitioning(const std::vector<int>& jumpsTracker)
	{
		segments.reserve(jumpsTracker.size());
		auto rightBound = static_cast<int>(jumpsTracker.size() - 1);
		while (true)
		{
			const auto leftBound = static_cast<int>(jumpsTracker.at(rightBound)) + 1;

			segments.push_back(Segment(leftBound, rightBound));
			if (leftBound == 0)
			{
				break;
			}
			rightBound = leftBound - 1;
		}
		std::reverse(segments.begin(), segments.end());
	}

	namespace
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
	}

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

	std::vector<double> computeOptimalEnergiesNoSegmentation(const Eigen::VectorXd& data, const int smoothnessOrder, const double smoothnessPenalty, const GivensCoefficients& givensCoeffs)
	{
		const int dataLength = static_cast<int>(data.size());
		assert(dataLength > 0);
		std::vector<double> smoothApproximationErrors(dataLength, 0);
		smoothApproximationErrors.resize(dataLength);

		auto intervalFromStart = Interval(0, data(0), smoothnessOrder, smoothnessPenalty);
		for (int idx = 1; idx < dataLength; idx++)
		{
			intervalFromStart.addNewDataPoint(givensCoeffs, data(idx));
			smoothApproximationErrors[idx] = intervalFromStart.approxError;
		}
		return smoothApproximationErrors;
	}


	/// @brief Aux functions for fct findBestPartition
	namespace
	{
		bool updateIntervalToRightBoundOrEraseIt(Interval& interval, const std::vector<double>& optimalEnergies,
			const Eigen::VectorXd& data, const int rightBound, const GivensCoefficients& givensCoeffs)
		{
			while (interval.rightBound < rightBound)
			{
				const auto currentRightBound = interval.rightBound;
				const auto eps_curr = interval.approxError;
				// Pruning strategy A: discard potential segments which can never be part of an optimal partitioning
				if (rightBound > 1 && currentRightBound < rightBound
					&& optimalEnergies[interval.leftBound - 1] + eps_curr >= optimalEnergies[currentRightBound])
				{
					return true;
				}
				else
				{
					// Extend current interval by new data and update its approximation error with Givens rotations
					interval.addNewDataPoint(givensCoeffs, data[interval.rightBound + 1]);
				}
			}
			return false;
		}

		void updateOptimalEnergyForRightBound(std::vector<double>& optimalEnergies, std::vector<int>& jumpsTracker, const Interval& interval, const int dataRightBound, const double jumpPenalty)
		{
			// Check if the current interval yields an improved energy value
			const auto optimalEnergyCandidate = optimalEnergies[interval.leftBound - 1] + jumpPenalty + interval.approxError;
			if (optimalEnergyCandidate <= optimalEnergies[dataRightBound])
			{
				optimalEnergies[dataRightBound] = optimalEnergyCandidate;
				jumpsTracker[dataRightBound] = interval.leftBound - 1;
			}
		}
	}

	Partitioning findBestPartition(Eigen::VectorXd& data, const int smoothnessOrder, const double smoothnessPenalty, const double jumpPenalty, const GivensCoefficients& givensCoeffs)
	{
		const auto dataLength = static_cast<int>(data.size());
		const auto optimalEnergiesNoJump = computeOptimalEnergiesNoSegmentation(data, smoothnessOrder, smoothnessPenalty, givensCoeffs);

		// vector with optimal functional values for each discrete interval [1..r], r = 1..n
		std::vector<double> optimalEnergies(dataLength, 0);

		// Keep track of the optimal segment boundaries by storing the optimal last jumps for each 
		// subdata on domains [1..r], r=1..dataLength
		std::vector<int> jumpsTracker(dataLength, -1);

		// container for the candidate segments, i.e., discrete intervals
		std::list<Interval> segments; // note: erasing from the middle of a list is cheaper than from a vector
		segments.push_back(Interval(1, data(1), smoothnessOrder, smoothnessPenalty));

		for (int dataRightBound = 1; dataRightBound < dataLength; dataRightBound++)
		{
			// Init with approximation error of single-segment partition, i.e. [0..dataRightBound]
			optimalEnergies[dataRightBound] = optimalEnergiesNoJump[dataRightBound];

			// Loop through candidate segments and find the best last jump for data[0..r-1]
			for (auto iter = segments.begin(); iter != segments.end();)
			{
				Interval& currInterval = *iter;
				// Update the current interval to match rightBound and apply pruning strategy A if applicable
				if (updateIntervalToRightBoundOrEraseIt(currInterval, optimalEnergies, data, dataRightBound, givensCoeffs))
				{
					iter = segments.erase(iter);
					continue;
				}
				else
				{
					iter++;
				}

				updateOptimalEnergyForRightBound(optimalEnergies, jumpsTracker, currInterval, dataRightBound, jumpPenalty);

				// Pruning strategy B: omit unnecessary computations of approximation errors
				if (currInterval.approxError + jumpPenalty > optimalEnergies[dataRightBound])
				{
					break;
				}
			}
			// Add the interval with left bound = rightBound to the list of segments
			if (dataRightBound < dataLength - 1)
			{
				segments.push_front(Interval(dataRightBound + 1, data(dataRightBound + 1), smoothnessOrder, smoothnessPenalty));
			}
		}

		return Partitioning(jumpsTracker);
	}


	Eigen::VectorXd computeResultsFromPartition(const Partitioning& partition, Eigen::VectorXd& data, const int smoothnessOrder, const double smoothnessPenalty, const GivensCoefficients& givensCoeffs)
	{
		const auto dataLength = static_cast<int>(data.size());
		Eigen::VectorXd pcwSmoothedResult = Eigen::VectorXd::Zero(dataLength);

		// Create interval objects corresponding to the segments of the partition
		std::vector<Interval> Intervals;
		Intervals.reserve(partition.size());
		for (const auto& segment : partition.segments)
		{
			const auto leftBound = segment.leftBound;
			const auto rightBound = segment.rightBound;
			const auto segmentSize = segment.size();

			if (segmentSize <= smoothnessOrder)
			{
				// nothing to do for small segments
				pcwSmoothedResult.segment(leftBound, segmentSize) = data.segment(leftBound, segmentSize);
				continue;
			}

			Intervals.push_back(Interval(leftBound, rightBound, smoothnessOrder, smoothnessPenalty, data.segment(leftBound, segmentSize)));
		};

		if (Intervals.empty())
		{
			// no reconstruction needed
			return pcwSmoothedResult;
		}

		// Sort Intervals in size-ascending order to avoid repeating identical row transformations of the system matrix
		std::sort(Intervals.begin(), Intervals.end(),
			[](const Interval& seg1, const Interval& seg2)
			{
				return seg1.size() < seg2.size();
			});

		// Solve the linear equation systems corresponding to each interval
		// Declare sparse system matrix of underlying least squares problem
		Eigen::MatrixXd systemMatrix = computeSystemMatrix(dataLength, smoothnessOrder, smoothnessPenalty);
		const auto isPcwPolynomial = std::isinf(smoothnessPenalty);

		// The "master" iterator knowing which intervals have to be considered (i.e. which interval lengths)
		auto startOfUnfinishedSegments = Intervals.begin();

		// Matrix elimination (i: row, j: column)
		int upperRowLength;
		int colOffset = 0;
		for (int i = 0; i < systemMatrix.rows(); i++)
		{
			if (!isPcwPolynomial && i < dataLength)
			{
				// nothing to eliminate yet
				continue;
			}

			for (int j = 0; j < smoothnessOrder + 1; j++)
			{
				if (isPcwPolynomial && (j == smoothnessOrder || i <= j))
				{
					break;
				}

				int rowOffset = 0;
				int lowerRowBeginCol = 0;
				if (isPcwPolynomial)
				{
					upperRowLength = smoothnessOrder;
				}
				else
				{
					rowOffset = dataLength - smoothnessOrder;
					upperRowLength = smoothnessOrder - j + 1;
					lowerRowBeginCol = j;
				}

				// Apply the Givens rotation to the corresponding matrix rows
				const double c = givensCoeffs.C(i - rowOffset, j);
				const double s = givensCoeffs.S(i - rowOffset, j);

				Eigen::MatrixXd upperMatRow = systemMatrix.block(j + colOffset, 0, 1, upperRowLength);
				Eigen::MatrixXd lowerMatRow = systemMatrix.block(i, lowerRowBeginCol, 1, upperRowLength);

				systemMatrix.block(j + colOffset, 0, 1, upperRowLength) = c * upperMatRow + s * lowerMatRow;
				systemMatrix.block(i, lowerRowBeginCol, 1, upperRowLength) = -s * upperMatRow + c * lowerMatRow;

				// Update interval data accordingly
				for (auto it = startOfUnfinishedSegments; it != Intervals.end(); it++)
				{
					Interval& currInterval = *it;
					currInterval.applyGivensRotationToData(givensCoeffs, i, j, rowOffset, colOffset);
				}
			}
			if (!isPcwPolynomial)
			{
				colOffset++;
			}



			// Fill result on finished intervals and delete them from the list of intervals
			//while (true)
			for (; startOfUnfinishedSegments != Intervals.end(); startOfUnfinishedSegments++)
			{
				Interval& currInterval = *startOfUnfinishedSegments;
				const auto currIntervalSize = currInterval.size();

				// If no interval has the current length, break:
				if ((isPcwPolynomial && (currIntervalSize != i + 1)) || (!isPcwPolynomial && (currIntervalSize != i - dataLength + smoothnessOrder + 1)))
				{
					break;
				}
				const auto leftBound = currInterval.leftBound;
				const auto rightBound = currInterval.rightBound;
				const auto& currData = currInterval.data;
				if (!isPcwPolynomial)
				{
					// Fill segment via back substitution
					pcwSmoothedResult(rightBound) = currData(currIntervalSize - 1) / systemMatrix(currIntervalSize - 1, 0);
					for (int ii = currIntervalSize - 2; ii >= 0; ii--)
					{
						double rhsSum = 0;
						for (int j = 1; j <= std::min(smoothnessOrder, currIntervalSize - ii - 1); j++)
						{
							rhsSum += systemMatrix(ii, j) * pcwSmoothedResult(leftBound + ii + j);
						}
						pcwSmoothedResult(leftBound + ii) = (currData(ii) - rhsSum) / systemMatrix(ii, 0);
					}
				}
				else
				{
					// Compute segment's polynomial coefficients
					const Eigen::VectorXd polynomialCoeffs = systemMatrix.block(0, 0, smoothnessOrder, smoothnessOrder).triangularView<Eigen::Upper>().solve(currData.head(smoothnessOrder));

					// Fill the segment with values induced by the polynomial coefficients
					pcwSmoothedResult.segment(leftBound, currIntervalSize).array() += polynomialCoeffs.tail(1).value();
					Eigen::VectorXd xValues = Eigen::VectorXd::LinSpaced(currIntervalSize, 1, currIntervalSize);
					for (int j = static_cast<int>(polynomialCoeffs.size()) - 2; j >= 0; j--)
					{
						pcwSmoothedResult.segment(leftBound, currIntervalSize) += polynomialCoeffs(j) * xValues;
						xValues.array() *= xValues.array();
					}
				}
			}

			// Stop when all segments are finished
			if (startOfUnfinishedSegments == Intervals.end())
			{
				return pcwSmoothedResult;
			}
		}
		return pcwSmoothedResult;
	}
}