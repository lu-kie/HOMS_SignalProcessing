#pragma once

#include "PcwSmoothPartitioningBase.h"
#include "HelperStructs.h"

#include <gtest/gtest.h>

namespace HOMS
{
	class PcwSmoothPartitioning : public PcwSmoothPartitioningBase
	{
	public:
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
		std::unique_ptr<ApproxIntervalBase> createIntervalForComputingPcwSmoothSignal(const int leftBound, const int rightBound, const Eigen::VectorXd& fullData) const;

		int m_smoothingOrder{ 1 };
		double m_smoothnessPenalty{ 1 };

		// unit tests
		FRIEND_TEST(ApproxIntervalSmooth, approxError);
		FRIEND_TEST(ApproxIntervalSmooth, applyGivensRotationToData);
		FRIEND_TEST(PcwSmoothPartitioning, createSystemMatrix);
		FRIEND_TEST(PcwSmoothPartitioning, computeGivensCoefficients);
		FRIEND_TEST(PcwSmoothPartitioning, computeOptimalEnergiesNoSegmentation);
		FRIEND_TEST(PcwSmoothPartitioning, findOptimalPartition);
	};
}