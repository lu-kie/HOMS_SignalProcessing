#pragma once

#include "PcwSmoothPartitioningBase.h"
#include "HelperStructs.h"
#include <gtest/gtest.h>

namespace HOMS
{
	class PcwPolynomialPartitioning : public PcwSmoothPartitioningBase
	{
	public:
		PcwPolynomialPartitioning(const int polynomialOrder, const double jumpPenalty, const int dataLength)
			: PcwSmoothPartitioningBase(jumpPenalty, dataLength)
			, m_polynomialOrder{ polynomialOrder }
		{
			if (m_polynomialOrder < 0)
			{
				throw std::invalid_argument("Requested polynomial order must be > 0");
			}
		}

	private:
		int minSegmentSize() const;
		void computeGivensCoefficients();
		Eigen::MatrixXd createSystemMatrix() const;
		void eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const;
		void fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::VectorXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const;
		std::unique_ptr<ApproxIntervalBase> createIntervalForPartitionFinding(const int leftBound, const double newDataPoint) const;
		std::unique_ptr<ApproxIntervalBase> createIntervalForComputingPcwSmoothSignal(const int leftBound, const int rightBound, const Eigen::VectorXd& data) const;

		int m_polynomialOrder{ 1 };

		// unit tests
		FRIEND_TEST(ApproxIntervalPolynomial, approxError);
		FRIEND_TEST(ApproxIntervalPolynomial, applyGivensRotationToData);
		FRIEND_TEST(PcwPolynomialPartitioning, createSystemMatrix);
		FRIEND_TEST(PcwPolynomialPartitioning, computeGivensCoefficients);
		FRIEND_TEST(PcwPolynomialPartitioning, computeOptimalEnergiesNoSegmentation);
		FRIEND_TEST(PcwPolynomialPartitioning, findOptimalPartition);
	};
}