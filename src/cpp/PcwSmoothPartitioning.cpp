#include "PcwSmoothPartitioning.h"

namespace homs
{
	int PcwSmoothPartitioning::minSegmentSize() const
	{
		return m_smoothingOrder + 1;
	}

	void PcwSmoothPartitioning::computeGivensCoefficients()
	{
		m_givensCoeffs.C = Eigen::MatrixXd::Zero(m_dataLength, m_smoothingOrder + 1);
		m_givensCoeffs.S = Eigen::MatrixXd::Zero(m_dataLength, m_smoothingOrder + 1);

		// Compute the coefficients of the Givens rotations to compute a QR decomposition of the systemMatrix.
		// Save them in C and S
		auto systemMatrix = createSystemMatrix();
		const auto numRows = static_cast<int>(systemMatrix.rows());
		const auto numCols = static_cast<int>(systemMatrix.cols());

		// compensating value for systemMatrix being stored sparsely (see member fct createSystemMatrix)
		const auto rowOffset = m_dataLength - m_smoothingOrder;
		for (int row = m_dataLength; row < numRows; row++)
		{

			for (int col = 0; col < numCols; col++)
			{
				// aux variables for compensating systemMatrix being stored sparsely (see member fct createSystemMatrix)
				const auto colOffset = row - m_dataLength;
				const auto upperRowLength = m_smoothingOrder - col + 1;
				const auto lowerRowBeginCol = col;

				// Determine Givens coefficients for eliminating systemMatrix(row,col) with Pivot element systemMatrix(row,row)
				auto rho = std::pow(systemMatrix(col + colOffset, 0), 2) + std::pow(systemMatrix(row, col), 2);
				rho = std::sqrt(rho);
				if (systemMatrix(col + colOffset, 0) < 0)
				{
					rho = -rho;
				}
				// store the coefficients
				m_givensCoeffs.C(row - rowOffset, col) = systemMatrix(col + colOffset, 0) / rho;
				m_givensCoeffs.S(row - rowOffset, col) = systemMatrix(row, col) / rho;
				// update the system matrix accordingly, i.e. apply the just computed Givens rotation to the corresponding matrix rows
				eliminateSystemMatrixEntry(systemMatrix, row, col);
			}
		}
	}

	namespace
	{
		/// @brief Compute the convolution of two vectors x,y
		/// @param x 
		/// @param y 
		/// @return convolution of x and y
		Eigen::VectorXd convolveVectors(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
		{
			const auto sizeX = x.size();
			const auto sizeY = y.size();
			Eigen::VectorXd conv = Eigen::VectorXd::Zero(sizeX + sizeY - 1);
			for (int j = 0; j < sizeY; j++)
			{
				for (int i = 0; i < sizeX; i++)
				{
					conv(j + i) += y(j) * x(i);
				}
			}
			return conv;
		}
	}

	Eigen::MatrixXd PcwSmoothPartitioning::createSystemMatrix() const
	{
		/* Example for k = 2, beta = 1
			systemMatrix = [ 1  0  0
							 ...
							 1  0  0
							 1 -2  1
							 ...
							 1 -2  1 ]

			which is a sparse representation of the full system matrix

			[1  0  0   ... 0
			 0  1  0   ... 0
			 ...
			 0  0  0   ... 1
			 1 -2  1   ... 0
			 0  1 -2 1 ... 0
			 ...
			 ...      1 -2 1 ]
			*/

		Eigen::MatrixXd systemMatrix = Eigen::MatrixXd::Zero(2 * m_dataLength - m_smoothingOrder, m_smoothingOrder + 1);
		// The upper block of systemMatrix is the identity matrix
		systemMatrix.col(0).setOnes();

		// The lower block has rows given by k-fold convolutions of the k-th order finite difference vector with itself
		Eigen::Vector2d forwardDifferenceCoeffs(-1, 1);
		Eigen::VectorXd kFoldFiniteDifferenceCoeffs(2);
		kFoldFiniteDifferenceCoeffs << -1, 1;
		for (int t = 0; t < m_smoothingOrder - 1; t++)
		{
			const auto convolutedCoeffs = convolveVectors(kFoldFiniteDifferenceCoeffs, forwardDifferenceCoeffs);
			kFoldFiniteDifferenceCoeffs.resizeLike(convolutedCoeffs);
			kFoldFiniteDifferenceCoeffs = convolutedCoeffs;
		}

		kFoldFiniteDifferenceCoeffs *= pow(m_smoothnessPenalty, m_smoothingOrder);
		for (int r = m_dataLength; r < 2 * m_dataLength - m_smoothingOrder; r++)
		{
			systemMatrix.row(r) = kFoldFiniteDifferenceCoeffs;
		}
		return systemMatrix;
	}

	void PcwSmoothPartitioning::eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const
	{
		if (row < m_dataLength)
		{
			return;
		}

		// aux variables for compensating systemMatrix being stored sparsely (see member fct createSystemMatrix)
		const auto rowOffset = m_dataLength - m_smoothingOrder;
		const auto colOffset = row - m_dataLength;
		const auto upperRowLength = m_smoothingOrder - col + 1;
		const auto lowerRowBeginCol = col;

		// Apply the Givens rotation to the corresponding matrix rows
		const double c = m_givensCoeffs.C(row - rowOffset, col);
		const double s = m_givensCoeffs.S(row - rowOffset, col);

		const Eigen::MatrixXd upperMatRow = systemMatrix.block(col + colOffset, 0, 1, upperRowLength);
		const Eigen::MatrixXd lowerMatRow = systemMatrix.block(row, lowerRowBeginCol, 1, upperRowLength);

		systemMatrix.block(col + colOffset, 0, 1, upperRowLength) = c * upperMatRow + s * lowerMatRow;
		systemMatrix.block(row, lowerRowBeginCol, 1, upperRowLength) = -s * upperMatRow + c * lowerMatRow;
	}

	void PcwSmoothPartitioning::fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::MatrixXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const
	{
		const auto leftBound = segment->leftBound;
		const auto rightBound = segment->rightBound;
		const auto& currData = segment->data;
		const auto segmentSize = segment->size();

		// Fill segment via back substitution
		resultToBeFilled.col(rightBound) = currData.col(segmentSize - 1) / partialUpperTriMat(segmentSize - 1, 0);
		for (int ii = segmentSize - 2; ii >= 0; ii--)
		{
			Eigen::VectorXd rhsSum = Eigen::VectorXd::Zero(m_numChannels);
			for (int j = 1; j <= std::min(m_smoothingOrder, segmentSize - ii - 1); j++)
			{
				rhsSum += partialUpperTriMat(ii, j) * resultToBeFilled.col(leftBound + ii + j);
			}
			resultToBeFilled.col(leftBound + ii) = (currData.col(ii) - rhsSum) / partialUpperTriMat(ii, 0);
		}
	}

	std::unique_ptr<ApproxIntervalBase> PcwSmoothPartitioning::createIntervalForPartitionFinding(const int leftBound, const Eigen::VectorXd&& newDataPoint) const
	{
		return std::make_unique<ApproxIntervalSmooth>(ApproxIntervalSmooth(leftBound, newDataPoint, m_smoothingOrder, m_smoothnessPenalty, m_numChannels));
	}

	std::unique_ptr<ApproxIntervalBase> PcwSmoothPartitioning::createIntervalForComputingResult(const int leftBound, const int rightBound, const Eigen::MatrixXd& fullData) const
	{

		return std::make_unique<ApproxIntervalSmooth>(ApproxIntervalSmooth(leftBound, rightBound, fullData, m_smoothingOrder, m_smoothnessPenalty));
	}
}