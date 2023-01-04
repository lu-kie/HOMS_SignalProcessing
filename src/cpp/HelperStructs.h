#pragma once
#include <Eigen/Dense>

namespace homs
{
	/// @brief Type of regularization
	enum class PcwRegularizationType
	{
		pcwSmooth,
		pcwPolynomial
	};

	/// @brief Givens coefficients c,s for computing a QR decomposition of a matrix
	struct GivensCoefficients
	{
		Eigen::MatrixXd C{};
		Eigen::MatrixXd S{};
	};

	/// @brief Single segment of a partitioning
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

	/// @brief Discrete partitioning of the domain 0,1,...,n in segments [leftBound, rightBound]
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
		/// @brief Constructor of base interval
		/// @param leftBound 
		/// @param rightBound 
		ApproxIntervalBase(const int leftBound, const int rightBound)
			: leftBound(leftBound)
			, rightBound(rightBound)
		{}

		/// @brief Constructor of base interval for given data and left boundary
		/// @param leftBound 
		/// @param data 
		ApproxIntervalBase(const int leftBound, Eigen::MatrixXd&& data)
			: leftBound(leftBound)
			, rightBound(leftBound + static_cast<int>(data.cols()) - 1)
			, data(data)
		{}

		/// @brief Constructor for finding an optimal partition
		/// @param leftBound 
		/// @param dataPoint 
		/// @param storedDataLength 
		/// @param numChannels
		/// @param regularization type of regularization in the interval
		ApproxIntervalBase(const int leftBound, const Eigen::VectorXd& dataPoint, const int storedDataLength, const int numChannels, const PcwRegularizationType regularization)
			: leftBound(leftBound)
			, rightBound(leftBound)
			, data(Eigen::MatrixXd::Zero(numChannels, storedDataLength))
		{
			switch (regularization)
			{
			case PcwRegularizationType::pcwPolynomial:
				data.col(0) = dataPoint;
				break;
			case PcwRegularizationType::pcwSmooth:
				data.col(storedDataLength - 1) = dataPoint;
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
		virtual void addNewDataPoint(const GivensCoefficients& givensCoeffs, Eigen::VectorXd&& newDataPoint) = 0;

		/// @brief Update associated data i.e. sparse Givens rotate it (for pcw. smoothed signal reconstruction)
		/// @param givensCoeffs Givens coefficients needed for the rotation
		/// @param row matrix row
		/// @param col matrix column
		virtual void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col) = 0;

		int leftBound{ -1 };			  ///< left bound of discrete interval
		int rightBound{ -1 };			  ///< right bound of discrete interval
		double approxError{ 0.0 };		  ///< optimal approximation error within interval
		Eigen::MatrixXd data{};			  ///< data corresponding to the interval
	};

	struct ApproxIntervalPolynomial : public ApproxIntervalBase
	{
		/// @brief Constructor of a polynomial interval: single data point
		/// @param leftBound 
		/// @param dataPoint 
		/// @param polynomialOrder 
		/// @param numChannels
		ApproxIntervalPolynomial(const int leftBound, const Eigen::VectorXd& dataPoint, const int polynomialOrder, const int numChannels)
			: ApproxIntervalBase(leftBound, dataPoint, polynomialOrder, numChannels, PcwRegularizationType::pcwPolynomial)
			, polynomialOrder(polynomialOrder)
		{}

		/// @brief Constructor of a polynomial interval from full data and interval boundaries
		/// @param leftBound 
		/// @param rightBound 
		/// @param fullData 
		/// @param polynomialOrder 
		ApproxIntervalPolynomial(const int leftBound, const int rightBound, const Eigen::MatrixXd& fullData, const int polynomialOrder)
			: ApproxIntervalBase(leftBound, Eigen::MatrixXd(fullData.middleCols(leftBound, rightBound - leftBound + 1)))
			, polynomialOrder(polynomialOrder)
		{}

		void addNewDataPoint(const GivensCoefficients& givensCoeffs, Eigen::VectorXd&& newDataPoint);
		void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col);

		int polynomialOrder{ 1 }; ///< order of the polynomial on the interval (1: constant, 2: linear etc.)
	};

	struct ApproxIntervalSmooth : public ApproxIntervalBase
	{
		/// @brief Constructor of a smooth interval: single data point
		/// @param leftBound 
		/// @param dataPoint 
		/// @param smoothingOrder 
		/// @param numChannels
		ApproxIntervalSmooth(const int leftBound, const Eigen::VectorXd& dataPoint, const int smoothingOrder, const int numChannels)
			: ApproxIntervalBase(leftBound, dataPoint, smoothingOrder, numChannels, PcwRegularizationType::pcwSmooth)
			, smoothingOrder(smoothingOrder)
		{}

		/// @brief Constructor of a smooth interval from full data and interval boundaries
		/// @param leftBound 
		/// @param rightBound 
		/// @param fullData 
		/// @param smoothingOrder 
		/// @param numChannels
		ApproxIntervalSmooth(const int leftBound, const int rightBound, const Eigen::MatrixXd& fullData, const int smoothingOrder)
			: ApproxIntervalBase(leftBound, rightBound)
			, smoothingOrder(smoothingOrder)
		{
			// For the reconstruction process the data vector must be appended by zeros for piecewise smooth reconstruction
			data = Eigen::MatrixXd::Zero(fullData.rows(), 2 * size() - smoothingOrder);
			data.leftCols(size()) = fullData.middleCols(leftBound, size());
		}

		void addNewDataPoint(const GivensCoefficients& givensCoeffs, Eigen::VectorXd&& newDataPoint);
		void applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col);

		int smoothingOrder{ 1 }; ///< order of discrete smoothness on the interval (1: first forward differences, 2: second order differences etc.)
	};

}