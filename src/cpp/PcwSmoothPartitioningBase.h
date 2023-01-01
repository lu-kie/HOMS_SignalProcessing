#pragma once
#include "HelperStructs.h"
#include <Eigen/Dense>

namespace HOMS
{
	class PcwSmoothPartitioningBase
	{
	public:
		/// @brief 
		/// @param jumpPenalty 
		/// @param dataLength 
		PcwSmoothPartitioningBase(const double jumpPenalty, const int dataLength)
			: m_dataLength{ dataLength }
			, m_jumpPenalty{ jumpPenalty }
		{
			if (m_dataLength <= 0)
			{
				throw std::invalid_argument("Requested data length must be > 0");
			}
			if (jumpPenalty < 0)
			{
				throw std::invalid_argument("Requested jump penalty must be >= 0");
			}
		}

		/// @brief 
		/// @param jumpPenalty 
		/// @param dataLength 
		/// @param givensCoeffs 
		PcwSmoothPartitioningBase(const double jumpPenalty, const int dataLength, const GivensCoefficients& givensCoeffs)
			: PcwSmoothPartitioningBase(jumpPenalty, dataLength)
		{
			m_givensCoeffs = givensCoeffs;
			m_initialized = true;
		}

		/// @brief Initializes the Givens coefficients. 
		/// Is automatically called when data comes in and the coefficients haven't been initialized.
		/// It is recommended to call this fct. from outside if several data sets shall be processed in parallel.
		void initialize();

		/// @brief Apply partitioning and pcw. smoothing to data
		/// @param data 
		/// @return optimal partitioning and corresponding pcw. smoothed signal
		std::pair<Eigen::VectorXd, Partitioning> applyToData(Eigen::VectorXd& data);

	protected:
		/// @brief Get the smallest size of a normal partitioning's segment
		/// @return 
		virtual int minSegmentSize() const = 0;

		/// @brief 
		/// @return 
		virtual Eigen::MatrixXd createSystemMatrix() const = 0;

		/// @brief 
		/// @return 
		virtual void computeGivensCoefficients() = 0;

		/// @brief 
		/// @param leftBound 
		/// @param newDataPoint 
		/// @return 
		virtual std::unique_ptr<ApproxIntervalBase> createIntervalForPartitionFinding(const int leftBound, const double newDataPoint) const = 0;

		/// @brief 
		/// @param leftBound 
		/// @param newDataPoint 
		/// @return 
		virtual std::unique_ptr<ApproxIntervalBase> createIntervalForComputingPcwSmoothSignal(const int leftBound, const int rightBound, const Eigen::VectorXd& data) const = 0;

		/// @brief 
		/// @param segment 
		/// @param resultToBeFilled 
		/// @param partialUpperTriMat 
		virtual void fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::VectorXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const = 0;

		/// @brief Eliminate an entry of the system matrix
		/// @param systemMatrix 
		/// @param row row index of the entry
		/// @param col col index of the entry
		virtual void eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const = 0;

		/// @brief 
		/// @param data 
		/// @return 
		std::vector<double> computeOptimalEnergiesNoSegmentation(const Eigen::VectorXd& data) const;

		/// @brief 
		/// @param data 
		/// @return 
		Partitioning findOptimalPartition(Eigen::VectorXd& data) const;

	protected:
		int m_dataLength{ 0 }; ///<
		GivensCoefficients m_givensCoeffs{}; ///<

	private:
		/// @brief 
		/// @param partition 
		/// @param minSegmentSize
		/// @param data 
		/// @param pcwPolynomialResult 
		/// @param polynomialOrder 
		/// @return 
		std::vector<std::unique_ptr<ApproxIntervalBase>> createIntervalsFromPartitionAndFillShortSegments(const Partitioning& partition, const int minSegmentSize, const Eigen::VectorXd& data, Eigen::VectorXd& resultToBeFilled) const;

		/// @brief 
		/// @param partition 
		/// @param data 
		/// @return 
		Eigen::VectorXd computePcwSmoothedSignalFromPartitioning(const Partitioning& partition, Eigen::VectorXd& data) const;

		bool m_initialized{ false };
		double m_jumpPenalty{ std::numeric_limits<double>::infinity() };
	};
}