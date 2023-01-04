#pragma once
#include "HelperStructs.h"
#include <Eigen/Dense>

namespace homs
{
	class PcwSmoothPartitioningBase
	{
	public:
		/// @brief Create base partitioning object with specified jump penalty and data length
		/// @param jumpPenalty 
		/// @param dataLength 
		PcwSmoothPartitioningBase(const double jumpPenalty, const int dataLength, const int numChannels)
			: m_dataLength(dataLength)
			, m_numChannels(numChannels)
			, m_jumpPenalty(jumpPenalty)
		{
			if (m_dataLength <= 0 || m_numChannels <= 0)
			{
				throw std::invalid_argument("Requested data length and number of channels must be > 0");
			}
			if (jumpPenalty < 0)
			{
				throw std::invalid_argument("Requested jump penalty must be >= 0");
			}
		}

		/// @brief Initializes the Givens coefficients. 
		/// Is automatically called when data comes in and the coefficients haven't been initialized.
		/// It is recommended to call this fct. from outside if several data sets shall be processed in parallel.
		void initialize();

		/// @brief Apply the partitioning and the corresponding piecewise smoothing to data
		/// @param data 
		/// @return optimal partitioning and corresponding pcw. smoothed signal
		std::pair<Eigen::MatrixXd, Partitioning> applyToData(Eigen::MatrixXd& data);

	protected:
		/// @brief Get the smallest size of a normal partitioning's segment
		/// @return min segment size
		virtual int minSegmentSize() const = 0;

		/// @brief Create the full system marix
		/// @return system matrix
		virtual Eigen::MatrixXd createSystemMatrix() const = 0;

		/// @brief Compute the Givens coefficients needed for a QR composition of the full system matrix
		virtual void computeGivensCoefficients() = 0;

		/// @brief Create a new interval object as needed in the dynamic programming scheme in findOptimalPartition
		/// @param leftBound left bound of the (single-point) interval
		/// @param newDataPoint data point corresponding to left bound
		/// @return interval object
		virtual std::unique_ptr<ApproxIntervalBase> createIntervalForPartitionFinding(const int leftBound, const Eigen::VectorXd&& newDataPoint) const = 0;

		/// @brief Create an interval object as needed in the smooth signal reconstruction process
		/// @param leftBound left bound of the interval
		/// @param rightBound right bound of the interval 
		/// @param data full data
		/// @return interval object
		virtual std::unique_ptr<ApproxIntervalBase> createIntervalForComputingResult(const int leftBound, const int rightBound, const Eigen::MatrixXd& data) const = 0;

		/// @brief Compute the best approximating smooth signal for the segment by performing back substition on the partial (full length) upper triangular system matrix
		/// @param segment 
		/// @param resultToBeFilled 
		/// @param partialUpperTriMat partial upper triangular system matrix which yields the best approximating signal on the given segment
		virtual void fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::MatrixXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const = 0;

		/// @brief Eliminate an entry of the system matrix
		/// @param systemMatrix 
		/// @param row row index of the entry
		/// @param col col index of the entry
		virtual void eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const = 0;

		/// @brief Compute the best approximation errors for data(0..r), r=1..dataLength
		/// @param data 
		/// @return best approximation errors
		std::vector<double> computeOptimalEnergiesNoSegmentation(const Eigen::MatrixXd& data) const;

		/// @brief Run the dynamic programming scheme to find an optimal partition of the input data
		/// @param data 
		/// @return optimal partition into (discrete) intervals
		Partitioning findOptimalPartition(Eigen::MatrixXd& data) const;

	protected:
		int m_dataLength{ 0 }; ///< number of data points of incoming data
		int m_numChannels{ 1 }; ///< number of channels of incoming data (e.g. 3 for data taken from an RGB image)
		GivensCoefficients m_givensCoeffs{}; ///< the Givens coefficients for obtaining a QR decomposition from the underlying system matrices. They further yield the recursion coefficients for the dynamic programming scheme

	private:

		/// @brief Create interval objects corresponding to the segments of the input partition. Trivially short segments will instead be filled immediately with data (resultToBeFilled)
		/// @param partition 
		/// @param minSegmentSize 
		/// @param data 
		/// @param resultToBeFilled 
		/// @return vector of intervals
		std::vector<std::unique_ptr<ApproxIntervalBase>> createIntervalsFromPartitionAndFillShortSegments(const Partitioning& partition, const int minSegmentSize, const Eigen::MatrixXd& data, Eigen::MatrixXd& resultToBeFilled) const;

		/// @brief Compute the best fitting piecewise smooth/polynomial result from the found optimal partition
		/// @param partition 
		/// @param data 
		/// @return piecewise smooth result
		Eigen::MatrixXd computePcwSmoothedSignalFromPartitioning(const Partitioning& partition, Eigen::MatrixXd& data) const;

		bool m_initialized{ false }; ///< flag if object is initialized, i.e. if the Givens coefficients have been computed
		double m_jumpPenalty{ std::numeric_limits<double>::infinity() }; ///< how much does introducing a new segment cost: large values give few segments and vice versa
	};
}