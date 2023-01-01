#include "PcwSmoothPartitioningBase.h"
#include "HelperStructs.h"

namespace HOMS
{
	std::pair<Eigen::VectorXd, Partitioning> PcwSmoothPartitioningBase::applyToData(Eigen::VectorXd& data)
	{
		if (data.size() != m_dataLength)
		{
			throw std::invalid_argument("Data size must fit implementation's data length");
		}

		if (!m_initialized)
		{
			initialize();
		}
		const auto optimalPartitioning = findOptimalPartition(data);
		Eigen::VectorXd pcwSmoothedResult = computePcwSmoothedSignalFromPartitioning(optimalPartitioning, data);
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

	std::vector<double> PcwSmoothPartitioningBase::computeOptimalEnergiesNoSegmentation(const Eigen::VectorXd& data) const
	{
		std::vector<double> approximationErrorsFromStart(m_dataLength, 0);
		approximationErrorsFromStart.resize(m_dataLength);

		auto intervalFromStart = createIntervalForPartitionFinding(0, data(0));
		for (int idx = 1; idx < m_dataLength; idx++)
		{
			intervalFromStart->addNewDataPoint(m_givensCoeffs, data(idx));
			approximationErrorsFromStart[idx] = intervalFromStart->approxError;
		}
		return approximationErrorsFromStart;

	}

	/// @brief Aux functions for fct findBestPartition
	namespace
	{
		bool updateIntervalToRightBoundOrEraseIt(ApproxIntervalBase& interval, const std::vector<double>& optimalEnergies,
			const Eigen::VectorXd& data, const int rightBound, const GivensCoefficients& givensCoeffs)
		{
			while (interval.rightBound < rightBound)
			{
				const auto currentRightBound = interval.rightBound;
				// Pruning strategy A: discard potential segments which can never be part of an optimal partitioning
				if (rightBound > 1 && currentRightBound < rightBound
					&& optimalEnergies[interval.leftBound - 1] + interval.approxError >= optimalEnergies[currentRightBound])
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

		void updateOptimalEnergyForRightBound(std::vector<double>& optimalEnergies, std::vector<int>& jumpsTracker, const ApproxIntervalBase& interval, const int dataRightBound, const double jumpPenalty)
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

	Partitioning PcwSmoothPartitioningBase::findOptimalPartition(Eigen::VectorXd& data) const
	{
		const auto approximationErrorsFromStart = computeOptimalEnergiesNoSegmentation(data);

		// vector with optimal functional values for each discrete interval [0..r], r = 0..n-1
		std::vector<double> optimalEnergies(m_dataLength, 0);

		// Keep track of the optimal segment boundaries by storing the optimal last left segment boundary for each 
		// subdata on domains [0..r], r = 0..n-1
		std::vector<int> jumpsTracker(m_dataLength, -1);

		// container for the candidate segments, i.e., discrete intervals
		std::list<std::unique_ptr<ApproxIntervalBase>> segments; // note: erasing from the middle of a list is cheaper than from a vector
		segments.push_back(createIntervalForPartitionFinding(1, data(1)));

		for (int dataRightBound = 1; dataRightBound < m_dataLength; dataRightBound++)
		{
			// Init with approximation error of single-segment partition, i.e. [0..dataRightBound], best last jump = 0
			optimalEnergies[dataRightBound] = approximationErrorsFromStart[dataRightBound];

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
					iter++;
				}

				updateOptimalEnergyForRightBound(optimalEnergies, jumpsTracker, currInterval, dataRightBound, m_jumpPenalty);

				// Pruning strategy B: omit unnecessary computations of approximation errors
				if (currInterval.approxError + m_jumpPenalty > optimalEnergies[dataRightBound])
				{
					break;
				}
			}
			// Add the interval with left bound = rightBound to the list of segments
			if (dataRightBound < m_dataLength - 1)
			{
				segments.push_front(createIntervalForPartitionFinding(dataRightBound + 1, data(dataRightBound + 1)));
			}
		}

		return Partitioning(jumpsTracker);
	}

	std::vector<std::unique_ptr<ApproxIntervalBase>> PcwSmoothPartitioningBase::createIntervalsFromPartitionAndFillShortSegments(const Partitioning& partition, const int minSegmentSize, const Eigen::VectorXd& data, Eigen::VectorXd& resultToBeFilled) const
	{
		std::vector<std::unique_ptr<ApproxIntervalBase>> Intervals;
		Intervals.reserve(partition.size());
		for (const auto& segment : partition.segments)
		{
			const auto leftBound = segment.leftBound;
			const auto rightBound = segment.rightBound;
			const auto segmentSize = segment.size();

			if (segmentSize < minSegmentSize)
			{
				// nothing to do for small segments
				resultToBeFilled.segment(leftBound, segmentSize) = data.segment(leftBound, segmentSize);
				continue;
			}
			Intervals.push_back(createIntervalForComputingPcwSmoothSignal(leftBound, rightBound, data));
		};

		// sort in size-ascending order
		std::sort(Intervals.begin(), Intervals.end(),
			[](const auto& seg1, const auto& seg2)
			{
				return seg1->size() < seg2->size();
			});

		return Intervals;
	}

	Eigen::VectorXd PcwSmoothPartitioningBase::computePcwSmoothedSignalFromPartitioning(const Partitioning& partition, Eigen::VectorXd& data) const
	{
		Eigen::VectorXd pcwSmoothResult = Eigen::VectorXd::Zero(m_dataLength);

		// Create interval objects corresponding to the segments of the partition
		// The intervals are sorted in size-ascending order to avoid repeating identical row transformations of the system matrix
		auto Intervals = createIntervalsFromPartitionAndFillShortSegments(partition, minSegmentSize(), data, pcwSmoothResult);

		if (Intervals.empty())
		{
			// no reconstruction needed
			return pcwSmoothResult;
		}

		// Solve the linear equation systems corresponding to each interval
		// Declare sparse system matrix of underlying least squares problem
		auto systemMatrix = createSystemMatrix();

		// The iterator knowing which intervals have to be considered (i.e. which interval lengths)
		auto beginOfUnfinishedSegments = Intervals.begin();

		// Matrix elimination (i: row, j: column)
		for (int row = 0; row < systemMatrix.rows(); row++)
		{
			for (int col = 0; col < systemMatrix.cols(); col++)
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
					// None of the unfinished intervals has the current length
					break;
				}
				fillSegmentFromPartialUpperTriangularSystemMatrix(currInterval, pcwSmoothResult, systemMatrix);
			}

			// Stop when all segments are finished
			if (beginOfUnfinishedSegments == Intervals.end())
			{
				return pcwSmoothResult;
			}
		}
		return pcwSmoothResult;
	}
}