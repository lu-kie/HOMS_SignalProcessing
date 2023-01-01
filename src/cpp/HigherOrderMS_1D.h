#pragma once
#include <Eigen/Dense>

namespace HOMS
{
	enum class PcwRegularizationType
	{
		pcwSmooth,
		pcwPolynomial
	};

	struct GivensCoefficients
	{
		GivensCoefficients()
		{
		}

		/// @brief Construct the Givens coefficients for computing a QR decomposition of the system matrix 
		/// corresponding to the smoothingOrder and smoothnessPenalty.
		/// @param dataLength 
		/// @param smoothingOrder 
		/// @param smoothnessPenalty 
		/// @param systemMatrix
		GivensCoefficients(const int dataLength, const int smoothingOrder, const double smoothnessPenalty);

		/// @brief Construct the Givens coefficients for computing a QR decomposition of a given system matrix 
		/// @param dataLength 
		/// @param smoothingOrder 
		/// @param smoothnessPenalty 
		/// @param systemMatrix
		GivensCoefficients(const int dataLength, const int smoothingOrder, const double smoothnessPenalty, Eigen::MatrixXd&& systemMatrix);

		Eigen::MatrixXd C{};
		Eigen::MatrixXd S{};
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
	/// @param smoothingOrder 
	/// @param smoothnessPenalty 
	/// @return the (sparse) system matrix
	Eigen::MatrixXd computeSystemMatrix(const int dataLength, const int smoothingOrder, const double smoothnessPenalty);
}