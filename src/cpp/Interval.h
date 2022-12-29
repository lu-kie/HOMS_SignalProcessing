#pragma once
#include "HigherOrderMS_1D.h"
#include <Eigen/Dense>

namespace HOMS
{
	struct Interval
	{
		/// @brief Constructor for finding an optimal partition
		/// @param leftBound 
		/// @param rightBound 
		/// @param dataPoint 
		/// @param smoothnessOrder 
		/// @param smoothnessPenalty 
		Interval(const int leftBound, const double dataPoint, const int smoothnessOrder, const double smoothnessPenalty)
			: leftBound(leftBound)
			, rightBound(leftBound)
			, smoothnessOrder(smoothnessOrder)
			, smoothnessPenalty(smoothnessPenalty)
		{
			data = Eigen::VectorXd::Zero(smoothnessOrder);
			if (std::isinf(smoothnessPenalty))
			{
				data(0) = dataPoint;
			}
			else
			{
				data(smoothnessOrder - 1) = dataPoint;
			}
		}

		/// @brief Constructor for the signal reconstruction from a partition
		/// @param leftBound 
		/// @param rightBound 
		/// @param smoothnessOrder 
		/// @param smoothnessPenalty 
		/// @param intervalData 
		Interval(const int leftBound, const int rightBound, const int smoothnessOrder, const double smoothnessPenalty, const Eigen::VectorXd& intervalData)
			: leftBound(leftBound)
			, rightBound(rightBound)
			, smoothnessOrder(smoothnessOrder)
			, smoothnessPenalty(smoothnessPenalty)
		{
			const auto intervalLength = size();
			if (intervalData.size() != intervalLength)
			{
				throw std::invalid_argument("Data and interval must have the same size.");
			}
			if (std::isinf(smoothnessPenalty))
			{
				data = intervalData;
			}
			else
			{
				// For the reconstruction process the data vector y must be appended by zeros for piecewise smooth reconstruction
				data = Eigen::VectorXd::Zero(2 * intervalLength - smoothnessOrder);
				data.head(intervalLength) = intervalData;
			}
		}

		/// @brief Give interval / data length
		/// @return interval length 
		int size() const;

		/// @brief Append data point to the interval and update the corresp. approximation error. 
		/// The interval is enlarged by one element.
		/// @param smoothnessOrder
		/// @param givensCoeffs 
		/// @param newDataPoint 
		void addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint);

		/// @brief Update associated data i.e. sparse Givens rotate it (for reconstruction process)
		/// @param givensCoeffs 
		/// @param row 
		/// @param col 
		/// @param rowOffset 
		/// @param colOffset
		void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col, const int rowOffset, const int colOffset);

		/// @brief 
		/// @param givensCoeffs 
		/// @param row 
		/// @param col 
		void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col);

		int leftBound{ -1 };			  ///< left bound of discrete interval
		int rightBound{ -1 };			  ///< right bound of discrete interval
		double approxError{ 0.0 };		  ///< optimal approximation error within interval
		int smoothnessOrder{ -1 };		  ///< smoothing order
		double smoothnessPenalty{ -1.0 }; ///< smoothing penalty
		Eigen::VectorXd data;			  ///< data corresponding to the interval
	};
}