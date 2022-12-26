#pragma once
#include <Eigen/Dense>

namespace HOMS
{
	struct GivensCoefficients
	{
		GivensCoefficients(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty);

		Eigen::MatrixXd C;
		Eigen::MatrixXd S;
	};

	// Computes the recursion coefficients for the dynamic programming scheme
	Eigen::MatrixXd computeSystemMatrix(const int dataLength, const int smoothnessOrder, const double smoothnessPenalty);
	/*
	// Converter from pointers to armadillo objects
	void armadilloConverter(double* data_raw, const int n, const int k, const double gamma,
		const double beta, double* J_raw, double* u_raw);
	*/
	/*
	// Computes and stores the approximation errors for intervals [1,r] for all r
	vec compute1rErrors(vec& data, const int n, const int k, const double beta, mat& C, mat& S);
	// Computes the optimal univariate partitioning for data data
	void findBestPartition(vec& data, const int n, const double gamma, vec& eps_1r,
		const int k, const double beta, mat& C, mat& S, vec& J);
	// Computes the corresponding reconstruction for an optimal partition
	void reconstructionFromPartition(vec& J, vec& u, vec& data, const int n, const int k, const double beta, mat& C, mat& S);
	bool compare_intervLengths(Interval inter1, Interval inter2);
	*/
}