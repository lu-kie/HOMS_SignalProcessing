#pragma once

#include "PcwSmoothPartitioningBase.h"
#include "HelperStructs.h"
#include <gtest/gtest.h>

namespace homs
{
	class PcwSmoothPartitioning : public PcwSmoothPartitioningBase
	{
	public:
		/// @brief Constructor from partition model parameters
		/// @param smoothingOrder order of the (discrete) smoothing on each segment (1: forward differences, 2: second centered differences etc.)
		/// @param jumpPenalty costs for introducing new segments
		/// @param dataLength length of incoming data
		PcwSmoothPartitioning(const int smoothingOrder, const double smoothnessPenalty, const double jumpPenalty, const int dataLength)
			: PcwSmoothPartitioningBase(jumpPenalty, dataLength)
			, m_smoothingOrder{ smoothingOrder }
			, m_smoothnessPenalty{ smoothnessPenalty }
		{
			if (m_smoothingOrder < 0 || m_smoothnessPenalty < 0)
			{
				throw std::invalid_argument("Requested smoothing order and smoothness penalty must be > 0");
			}
		}

	private:
		int minSegmentSize() const;
		void computeGivensCoefficients();
		Eigen::MatrixXd createSystemMatrix() const;
		void eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const;
		void fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::VectorXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const;
		std::unique_ptr<ApproxIntervalBase> createIntervalForPartitionFinding(const int leftBound, const double newDataPoint) const;
		std::unique_ptr<ApproxIntervalBase> createIntervalForComputingResult(const int leftBound, const int rightBound, const Eigen::VectorXd& fullData) const;

		int m_smoothingOrder{ 1 }; ///< order of the piecewise (discrete) smooth partitioning and smoothing (1: first differences, 2: second order differences etc.)
		double m_smoothnessPenalty{ 1 }; ///< how much the smoothness is penalized, i.e. enforced (larger values enforce more smoothing, limit case is piecewise polynomial smoothing)

		// unit tests
		FRIEND_TEST(ApproxIntervalSmooth, approxError);
		FRIEND_TEST(ApproxIntervalSmooth, applyGivensRotationToData);
		FRIEND_TEST(PcwSmoothPartitioning, createSystemMatrix);
		FRIEND_TEST(PcwSmoothPartitioning, computeGivensCoefficients);
		FRIEND_TEST(PcwSmoothPartitioning, computeOptimalEnergiesNoSegmentation);
		FRIEND_TEST(PcwSmoothPartitioning, findOptimalPartition);
	};
}