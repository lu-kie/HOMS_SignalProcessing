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
		/// @brief Default ctor
		Partitioning() {};

		/// @brief Construct a partitioning object from the tracked last optimal jumps from findBestPartition
		/// @param jumpsTracker 
		Partitioning(const std::vector<int>& jumpsTracker);

		/// @brief get the number of segments
		/// @return number of segments
		int size() const { return static_cast<int>(segments.size()); };

		std::vector<Segment> segments; //< the segments of the partitioning encoded as left and right bounds
	};

	struct ApproxIntervalBase
	{
		ApproxIntervalBase(const int leftBound, const int rightBound)
			: leftBound(leftBound)
			, rightBound(rightBound)
		{}

		ApproxIntervalBase(const int leftBound, Eigen::VectorXd&& data)
			: leftBound(leftBound)
			, rightBound(leftBound + static_cast<int>(data.size()) - 1)
			, data(data)
		{}

		/// @brief Constructor for finding an optimal partition
		/// @param leftBound 
		/// @param dataPoint 
		/// @param storedDataSize 
		/// @param regularization 
		ApproxIntervalBase(const int leftBound, const double dataPoint, const int storedDataSize, const PcwRegularizationType regularization)
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

	struct ApproxIntervalPolynomial : public ApproxIntervalBase
	{
		ApproxIntervalPolynomial(const int leftBound, const double dataPoint, const int polynomialOrder)
			: ApproxIntervalBase(leftBound, dataPoint, polynomialOrder, PcwRegularizationType::pcwPolynomial)
			, polynomialOrder(polynomialOrder)
		{}

		ApproxIntervalPolynomial(const int leftBound, const int rightBound, const Eigen::VectorXd& fullData, const int polynomialOrder)
			: ApproxIntervalBase(leftBound, Eigen::VectorXd(fullData.segment(leftBound, rightBound - leftBound + 1)))
			, polynomialOrder(polynomialOrder)
		{}

		void addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint);
		void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col);

		int polynomialOrder{ 1 }; ///< order of the polynomial on the interval (1: constant, 2: linear etc.)
	};


	struct ApproxIntervalSmooth : public ApproxIntervalBase
	{
		ApproxIntervalSmooth(const int leftBound, const double dataPoint, const int smoothingOrder, const double smoothnessPenalty)
			: ApproxIntervalBase(leftBound, dataPoint, smoothingOrder, PcwRegularizationType::pcwSmooth)
			, smoothingOrder(smoothingOrder)
			, smoothnessPenalty(smoothnessPenalty)
		{}

		ApproxIntervalSmooth(const int leftBound, const int rightBound, const Eigen::VectorXd& fullData, const int smoothingOrder, const double smoothnessPenalty)
			: ApproxIntervalBase(leftBound, rightBound)
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