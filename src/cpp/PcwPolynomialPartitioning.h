#pragma once

#include "PcwSmoothPartitioningBase.h"
#include "HelperStructs.h"
#include <gtest/gtest.h>

namespace homs
{
	class PcwPolynomialPartitioning : public PcwSmoothPartitioningBase
	{
	public:
		/// @brief Constructor from partition model parameters
		/// @param polynomialOrder order of the polynomial on each segment
		/// @param jumpPenalty costs for introducing new segments
		/// @param dataLength length of incoming data
		/// @param numChannels number of channels of incoming data
		PcwPolynomialPartitioning(const int polynomialOrder, const double jumpPenalty, const int dataLength, const int numChannels)
			: PcwSmoothPartitioningBase(jumpPenalty, dataLength, numChannels)
			, m_polynomialOrder{ polynomialOrder }
			, m_fullSystemMatrixTr{createSystemMatrix().transpose()}
		{
			if (m_polynomialOrder <= 0)
			{
				throw std::invalid_argument("Requested polynomial order must be > 0");
			}
			m_givensCoeffs.C = Eigen::MatrixXd::Zero(m_dataLength, m_polynomialOrder);
			m_givensCoeffs.S = Eigen::MatrixXd::Zero(m_dataLength, m_polynomialOrder);
		}

	private:
		int minSegmentSize() const;
		void computeGivensCoefficients();
		Eigen::MatrixXd createSystemMatrix() const;
		void eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const;
		void fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::MatrixXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const;
		std::unique_ptr<ApproxIntervalBase> createIntervalForPartitionFinding(const int leftBound, const Eigen::Map<Eigen::MatrixXd>& fullData) const;
		std::unique_ptr<ApproxIntervalBase> createIntervalForComputingResult(const int leftBound, const int rightBound, const Eigen::Map<Eigen::MatrixXd>& fullData) const;

		const int m_polynomialOrder{ 1 }; ///< order of the piecewise polynomial partitioning and smoothing (1: constant, 2: affine linear etc.)
		const Eigen::MatrixXd m_fullSystemMatrixTr{m_dataLength, m_polynomialOrder}; ///< transposed full system matrix. Used for convenience when computing the result signal on the found partition
		// unit tests
		FRIEND_TEST(ApproxIntervalPolynomial, approxError);
		FRIEND_TEST(ApproxIntervalPolynomial, applyGivensRotationToData);
		FRIEND_TEST(PcwPolynomialPartitioning, createSystemMatrix);
		FRIEND_TEST(PcwPolynomialPartitioning, computeGivensCoefficients);
		FRIEND_TEST(PcwPolynomialPartitioning, computeOptimalEnergiesNoSegmentation);
		FRIEND_TEST(PcwPolynomialPartitioning, findOptimalPartition);
		FRIEND_TEST(PcwPolynomialPartitioning, computeSignalFromPartitioning);
	};
}