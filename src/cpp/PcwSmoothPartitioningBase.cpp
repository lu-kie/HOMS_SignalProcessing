#include "PcwSmoothPartitioningBase.h"
#include "HelperStructs.h"

namespace homs
{
	std::pair<Eigen::MatrixXd, Partitioning> PcwSmoothPartitioningBase::applyToData(Eigen::MatrixXd& data)
	{
		return applyToData(data.data());
	}

	std::pair<Eigen::MatrixXd, Partitioning> PcwSmoothPartitioningBase::applyToData(double* data)
	{
		const Eigen::Map<Eigen::MatrixXd> dataMap(data, m_numChannels, m_dataLength);
		return applyToData(dataMap);
	}

	std::pair<Eigen::MatrixXd, Partitioning> PcwSmoothPartitioningBase::applyToData(const Eigen::Map<Eigen::MatrixXd>& data)
	{
		if (static_cast<int>(data.cols()) != m_dataLength)
		{
			throw std::invalid_argument("Input data length must fit implementation's data length");
		}
		if (static_cast<int>(data.rows()) != m_numChannels)
		{
			throw std::invalid_argument("Input data's number of channels must fit implementation's number of channels");
		}

		if (!m_initialized)
		{
			initialize();
		}
		const auto optimalPartitioning = findOptimalPartition(data);
		Eigen::MatrixXd pcwSmoothedResult = computePcwSmoothedSignalFromPartitioning(optimalPartitioning, data);
		return std::make_pair(pcwSmoothedResult, optimalPartitioning);
	}

	void PcwSmoothPartitioningBase::initialize()
	{
		if (m_initialized)
		{
			return;
		}
		computeGivensCoefficients();
		m_initialized = true;
	}

	std::vector<double> PcwSmoothPartitioningBase::computeOptimalEnergiesNoSegmentation(const Eigen::Map<Eigen::MatrixXd>& data) const
	{
		std::vector<double> approximationErrorsFromStart(m_dataLength, 0);

		auto intervalFromStart = createIntervalForPartitionFinding(0, data.col(0));
		for (int rightBound = 1; rightBound < m_dataLength; rightBound++)
		{
			intervalFromStart->addNewDataPoint(m_givensCoeffs, data.col(rightBound));
			approximationErrorsFromStart[rightBound] = intervalFromStart->approxError;
		}

		return approximationErrorsFromStart;
	}

	/// @brief Aux functions for fct findBestPartition
	namespace
	{
		bool updateIntervalToRightBoundOrEraseIt(ApproxIntervalBase& interval, const std::vector<double>& optimalEnergies,
			const Eigen::Map<Eigen::MatrixXd>& data, const int rightBound, const GivensCoefficients& givensCoeffs)
		{
			while (interval.rightBound < rightBound)
			{
				// Extend current interval by new data and update its approximation error with Givens rotations
				interval.addNewDataPoint(givensCoeffs, data.col(interval.rightBound + 1));

				// Pruning strategy A: discard potential segments which can never be part of an optimal partitioning
				if (const auto currentRightBound = interval.rightBound;
					optimalEnergies[currentRightBound] <= optimalEnergies[interval.leftBound - 1] + interval.approxError)
				{
					return true;
				}
			}
			return false;
		}

		void updateOptimalEnergyForRightBound(std::vector<double>& optimalEnergies, std::vector<int>& jumpsTracker, const ApproxIntervalBase& interval, const int dataRightBound, const double jumpPenalty)
		{
			// Check if the current interval yields an improved energy value
			if (const auto optimalEnergyCandidate = optimalEnergies[interval.leftBound - 1] + jumpPenalty + interval.approxError;
				optimalEnergyCandidate <= optimalEnergies[dataRightBound])
			{
				optimalEnergies[dataRightBound] = optimalEnergyCandidate;
				jumpsTracker[dataRightBound] = interval.leftBound - 1;
			}
		}
	}

	Partitioning PcwSmoothPartitioningBase::findOptimalPartition(const Eigen::Map<Eigen::MatrixXd>& data) const
	{
		// container for storing the optimal functional values for each discrete interval [0..r], r = 0..n-1
		auto optimalEnergies = computeOptimalEnergiesNoSegmentation(data);

		// Keep track of the optimal segment boundaries by storing the optimal last left segment boundary for each 
		// subdata on domains [0..r], r = 0..n-1
		std::vector<int> jumpsTracker(m_dataLength, -1);

		// container for the candidate segments, i.e., discrete intervals
		std::list<std::unique_ptr<ApproxIntervalBase>> segments; // note: erasing from the middle of a list is cheaper than from a vector

		for (int dataRightBound = 1; dataRightBound < m_dataLength; dataRightBound++)
		{
			segments.push_front(createIntervalForPartitionFinding(dataRightBound, data.col(dataRightBound)));
			// Loop through candidate segments and find the best last jump for data[0..dataRightBound]
			for (auto iter = segments.begin(); iter != segments.end();)
			{
				auto& currInterval = *(*iter);
				// Update the current interval to match rightBound and apply pruning strategy A if applicable
				if (updateIntervalToRightBoundOrEraseIt(currInterval, optimalEnergies, data, dataRightBound, m_givensCoeffs))
				{
					iter = segments.erase(iter);
					continue;
				}
				else
				{
					updateOptimalEnergyForRightBound(optimalEnergies, jumpsTracker, currInterval, dataRightBound, m_jumpPenalty);

					// Pruning strategy B: omit unnecessary computations of approximation errors
					if (currInterval.approxError + m_jumpPenalty > optimalEnergies[dataRightBound])
					{
						break;
					}
					iter++;
				}
			}
		}

		return Partitioning(jumpsTracker);
	}

	std::vector<std::unique_ptr<ApproxIntervalBase>> PcwSmoothPartitioningBase::createIntervalsFromPartitionForComputingResult(
		const Partitioning& partition, const int minSegmentSize, const Eigen::Map<Eigen::MatrixXd>& data, Eigen::MatrixXd& resultToBeFilled) const
	{
		std::vector<std::unique_ptr<ApproxIntervalBase>> Intervals;
		Intervals.reserve(partition.size());
		for (const auto& segment : partition.segments)
		{
			const auto leftBound = segment.leftBound;
			const auto rightBound = segment.rightBound;

			if (const auto segmentSize = segment.size();
				segmentSize < minSegmentSize)
			{
				// nothing to do for small segments
				resultToBeFilled.middleCols(leftBound, segmentSize) = data.middleCols(leftBound, segmentSize);
			}
			else
			{
				Intervals.push_back(createIntervalForComputingResult(leftBound, rightBound, data));
			}
		};

		// sort in size-ascending order
		std::sort(Intervals.begin(), Intervals.end(),
			[](const auto& seg1, const auto& seg2)
			{
				return seg1->size() < seg2->size();
			});

		return Intervals;
	}

	Eigen::MatrixXd PcwSmoothPartitioningBase::computePcwSmoothedSignalFromPartitioning(const Partitioning& partition, const Eigen::Map<Eigen::MatrixXd>& data) const
	{
		Eigen::MatrixXd pcwSmoothResult = Eigen::MatrixXd::Zero(m_numChannels, m_dataLength);

		// Create interval objects corresponding to the segments of the partition
		// The intervals are sorted in size-ascending order to avoid repeating identical row transformations of the system matrix
		auto Intervals = createIntervalsFromPartitionForComputingResult(partition, minSegmentSize(), data, pcwSmoothResult);

		if (Intervals.empty())
		{
			// only short segments, result already reconstructed
			return pcwSmoothResult;
		}

		// Solve the linear equation systems corresponding to each interval for all intervals at once
		auto systemMatrix = createSystemMatrix();

		// The iterator knowing which intervals have to be considered (i.e. which interval lengths)
		auto beginOfUnfinishedSegments = Intervals.begin();

		// Matrix elimination (i: row, j: column)
		const auto numRows = static_cast<int>(systemMatrix.rows());
		const auto numCols = static_cast<int>(systemMatrix.cols());
		for (int row = 0; row < numRows; row++)
		{
			for (int col = 0; col < numCols; col++)
			{
				eliminateSystemMatrixEntry(systemMatrix, row, col);

				// Update interval data accordingly
				for (auto it = beginOfUnfinishedSegments; it != Intervals.end(); it++)
				{
					auto& currInterval = *it;
					currInterval->applyGivensRotationToData(m_givensCoeffs, row, col);
				}
			}

			// Fill result on finished intervals and delete them from the list of intervals
			for (; beginOfUnfinishedSegments != Intervals.end(); beginOfUnfinishedSegments++)
			{
				auto currInterval = (*beginOfUnfinishedSegments).get();

				if (const auto currIntervalSize = currInterval->size();
					currIntervalSize != row + 1)
				{
					// None of the unfinished intervals has the currently processed data length
					break;
				}
				fillSegmentFromPartialUpperTriangularSystemMatrix(currInterval, pcwSmoothResult, systemMatrix);
			}

			// All segments are finished
			if (beginOfUnfinishedSegments == Intervals.end())
			{
				return pcwSmoothResult;
			}
		}
		return pcwSmoothResult;
	}
}