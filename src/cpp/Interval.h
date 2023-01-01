#pragma once
#include "HigherOrderMS_1D.h"
#include <Eigen/Dense>

namespace HOMS
{
	struct Interval
	{
		/// @brief Constructor for finding an optimal partition
		/// @param leftBound  
		/// @param dataPoint 
		/// @param smoothingOrder 
		/// @param smoothnessPenalty 
		Interval(const int leftBound, const double dataPoint, const int smoothingOrder, const double smoothnessPenalty)
			: leftBound(leftBound)
			, rightBound(leftBound)
			, smoothingOrder(smoothingOrder)
			, smoothnessPenalty(smoothnessPenalty)
		{
			data = Eigen::VectorXd::Zero(smoothingOrder);
			if (std::isinf(smoothnessPenalty))
			{
				data(0) = dataPoint;
			}
			else
			{
				data(smoothingOrder - 1) = dataPoint;
			}
		}

		/// @brief Constructor for the signal reconstruction from a partition
		/// @param leftBound 
		/// @param rightBound 
		/// @param smoothingOrder 
		/// @param smoothnessPenalty 
		/// @param intervalData 
		Interval(const int leftBound, const int rightBound, const int smoothingOrder, const double smoothnessPenalty, const Eigen::VectorXd& intervalData)
			: leftBound(leftBound)
			, rightBound(rightBound)
			, smoothingOrder(smoothingOrder)
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
				data = Eigen::VectorXd::Zero(2 * intervalLength - smoothingOrder);
				data.head(intervalLength) = intervalData;
			}
		}

		/// @brief Give interval / data length
		/// @return interval length 
		int size() const;

		/// @brief Append data point to the interval and update the corresp. approximation error. 
		/// The interval is enlarged by one element.
		/// @param smoothingOrder
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
		int smoothingOrder{ -1 };		  ///< smoothing order
		double smoothnessPenalty{ -1.0 }; ///< smoothing penalty
		Eigen::VectorXd data;			  ///< data corresponding to the interval
	};

	struct IntervalBase
	{
		IntervalBase(const int leftBound, const int rightBound)
			: leftBound(leftBound)
			, rightBound(rightBound)
		{}

		IntervalBase(const int leftBound, Eigen::VectorXd&& data)
			: leftBound(leftBound)
			, rightBound(leftBound + static_cast<int>(data.size()) - 1)
			, data(data)
		{
		}

		/// @brief Constructor for finding an optimal partition
		/// @param leftBound 
		/// @param dataPoint 
		/// @param storedDataSize 
		/// @param regularization 
		IntervalBase(const int leftBound, const double dataPoint, const int storedDataSize, const PcwRegularizationType regularization)
			: leftBound(leftBound)
			, rightBound(leftBound)
			, data(Eigen::VectorXd::Zero(storedDataSize))
		{
			switch (regularization)
			{
			case PcwRegularizationType::pcwPolynomial:
				data(0) = dataPoint;
				break;
			case PcwRegularizationType::pcwSmooth:
				data(storedDataSize - 1) = dataPoint;
				break;
			default:
				throw std::invalid_argument("Unsupported pcw regularization type");
			}
		}

		/// @brief Give interval / data length
		/// @return interval length 
		int size() const { return rightBound - leftBound + 1; };

		/// @brief Append data point to the interval and update the corresp. approximation error. 
		/// The interval is enlarged by one element.
		/// @param givensCoeffs 
		/// @param newDataPoint 
		virtual void addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint) = 0;

		/// @brief Update associated data i.e. sparse Givens rotate it (for pcw. smoothed signal reconstruction)
		/// @param givensCoeffs 
		/// @param row 
		/// @param col 
		virtual void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col) = 0;

		int leftBound{ -1 };			  ///< left bound of discrete interval
		int rightBound{ -1 };			  ///< right bound of discrete interval
		double approxError{ 0.0 };		  ///< optimal approximation error within interval
		Eigen::VectorXd data{};			  ///< data corresponding to the interval
	};

	struct IntervalPolynomial : public IntervalBase
	{
		IntervalPolynomial(const int leftBound, const double dataPoint, const int polynomialOrder)
			: IntervalBase(leftBound, dataPoint, polynomialOrder, PcwRegularizationType::pcwPolynomial)
			, polynomialOrder(polynomialOrder)
		{
		}

		IntervalPolynomial(const int leftBound, const int rightBound, const Eigen::VectorXd& fullData, const int polynomialOrder)
			: IntervalBase(leftBound, Eigen::VectorXd(fullData.segment(leftBound, rightBound - leftBound + 1)))
			, polynomialOrder(polynomialOrder)
		{
		}

		void addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint);
		void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col);

		int polynomialOrder{ 1 }; ///< order of the polynomial on the interval (1: constant, 2: linear etc.)
	};


	struct IntervalSmooth : public IntervalBase
	{
		IntervalSmooth(const int leftBound, const double dataPoint, const int smoothingOrder, const double smoothnessPenalty)
			: IntervalBase(leftBound, dataPoint, smoothingOrder, PcwRegularizationType::pcwSmooth)
			, smoothingOrder(smoothingOrder)
			, smoothnessPenalty(smoothnessPenalty)
		{}

		IntervalSmooth(const int leftBound, const int rightBound, const Eigen::VectorXd& fullData, const int smoothingOrder, const double smoothnessPenalty)
			: IntervalBase(leftBound, rightBound)
			, smoothingOrder(smoothingOrder)
			, smoothnessPenalty(smoothnessPenalty)
		{
			// For the reconstruction process the data vector must be appended by zeros for piecewise smooth reconstruction
			data = Eigen::VectorXd::Zero(2 * size() - smoothingOrder);
			data.head(size()) = fullData.segment(leftBound, size());
		}

		void addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint);
		void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col);

		int smoothingOrder{ 1 }; ///< order of discrete smoothness on the interval (1: first forward differences, 2: second order differences etc.)
		double smoothnessPenalty{ 1 }; ///< how much is the smoothness enforced
	};
}