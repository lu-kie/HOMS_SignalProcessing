#pragma once
#include <Eigen/Dense>

namespace HOMS
{
	struct GivensCoefficients
	{
		/// @brief Construct the Givens coefficients for computing a QR decomposition of the system matrix 
		/// corresponding to the smoothnessOrder and smoothnessPenalty.
		/// @param dataLength 
		/// @param smoothnessOrder 
		/// @param smoothnessPenalty 
		GivensCoefficients(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty);

		Eigen::MatrixXd C;
		Eigen::MatrixXd S;
	};


	struct Segment
	{
		int size() const { return rightBound - leftBound + 1; }

		int leftBound{ 0 };
		int rightBound{ 0 };


		bool operator==(const Segment& rhs) const
		{
			return leftBound == rhs.leftBound && rightBound == rhs.rightBound;
		}
	};

	struct Partitioning
	{
		/// @brief Construct a partitioning object from the tracked last optimal jumps from findBestPartition
		/// @param jumpsTracker 
		Partitioning(const std::vector<int>& jumpsTracker);

		/// @brief Default ctor
		Partitioning() {};

		/// @brief get the number of segments
		/// @return number of segments
		int size() const { return static_cast<int>(segments.size()); };

		std::vector<Segment> segments; //< the segments of the partitioning encoded as left and right bounds
	};


	/// @brief Provides the underlying (sparse) system matrix for the given parameters
	/// @param dataLength 
	/// @param smoothnessOrder 
	/// @param smoothnessPenalty 
	/// @return the (sparse) system matrix
	Eigen::MatrixXd computeSystemMatrix(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty);


	/// @brief Computes the optimal approximation errors for the discrete intervals [1,r] for all r = 1...n
	/// @param data 
	/// @param smoothnessOrder 
	/// @param smoothnessPenalty 
	/// @param givensCoeffs 
	/// @return optimal single segment energies
	std::vector<double> computeOptimalEnergiesNoSegmentation(const Eigen::VectorXd& data, const int smoothnessOrder, const double smoothnessPenalty, const GivensCoefficients& givensCoeffs);

	/// @brief Computes the optimal partitioning for the input data
	/// @param data 
	/// @param smoothnessOrder 
	/// @param smoothnessPenalty 
	/// @param jumpPenalty 
	/// @param givensCoeffs 
	/// @return optimal partition encoded as pairs of segment boundaries
	Partitioning findBestPartition(Eigen::VectorXd& data, const int smoothnessOrder, const double smoothnessPenalty, const double jumpPenalty, const GivensCoefficients& givensCoeffs);

	/// @brief Compute the corresponding piecewised smoothed signal from an optimal partition
	/// @param partition 
	/// @param data 
	/// @param smoothnessOrder 
	/// @param smoothnessPenalty 
	/// @param givensCoeffs 
	/// @return 
	Eigen::VectorXd computeResultsFromPartition(const Partitioning& partition, Eigen::VectorXd& data, const int smoothnessOrder, const double smoothnessPenalty, const GivensCoefficients& givensCoeffs);
}