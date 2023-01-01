#pragma once

#include "PcwSmoothPartitioningBase.h"
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

		PcwPolynomialPartitioning(const int polynomialOrder, const double jumpPenalty, const int dataLength, const GivensCoefficients& givensCoeffs)
			: PcwSmoothPartitioningBase(jumpPenalty, dataLength, givensCoeffs)
			, m_polynomialOrder{ polynomialOrder }
		{
			if (m_polynomialOrder < 0)
			{
				throw std::invalid_argument("Requested polynomial order must be > 0");
			}

			if (givensCoeffs.C.rows() < m_dataLength)
			{
				throw std::invalid_argument("Provided Givens coefficients must be created for same or larger data length");
			}
		}


	private:
		int minSegmentSize() const;
		GivensCoefficients createGivensCoefficients() const;
		Eigen::MatrixXd computeSystemMatrix() const;
		void eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const;
		void fillSegmentFromPartialUpperTriangularSystemMatrix(IntervalBase* segment, Eigen::VectorXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const;
		std::unique_ptr<IntervalBase> createIntervalForPartitionFinding(const int leftBound, const double newDataPoint) const;
		std::unique_ptr<IntervalBase> createIntervalForComputingPcwSmoothSignal(const int leftBound, const int rightBound, const Eigen::VectorXd& data) const;

		int m_polynomialOrder{ 1 };

		// unit tests
		FRIEND_TEST(PcwPolynomialPartitioning, computeSystemMatrix);
		FRIEND_TEST(PcwPolynomialPartitioning, createGivensCoefficients);
		FRIEND_TEST(PcwPolynomialPartitioning, computeOptimalEnergiesNoSegmentation);
		FRIEND_TEST(PcwPolynomialPartitioning, findOptimalPartition);
	};
}