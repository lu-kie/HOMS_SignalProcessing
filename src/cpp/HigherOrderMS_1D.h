#pragma once
#include <Eigen/Dense>

namespace HOMS
{
	struct GivensCoefficients
	{
		/// @brief 
		/// @param dataLength 
		/// @param smoothnessOrder 
		/// @param smoothnessPenalty 
		GivensCoefficients(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty);

		Eigen::MatrixXd C;
		Eigen::MatrixXd S;
	};

	struct Partitioning
	{
		Partitioning(const std::vector<int>& jumpsTracker);
		int size() const { return static_cast<int>(segments.size()); };
		std::vector<std::pair<int, int>> segments;

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

	// Computes the corresponding reconstruction for an optimal partition
	/*
	void reconstructionFromPartition(vec& J, vec& u, vec& data, const int n, const int k, const double beta, mat& C, mat& S);
	bool compare_intervLengths(Interval inter1, Interval inter2);
	*/
}