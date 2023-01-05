#include "PcwPolynomialPartitioning.h"

namespace homs
{
	int PcwPolynomialPartitioning::minSegmentSize() const
	{
		return m_polynomialOrder + 1;
	}

	void PcwPolynomialPartitioning::computeGivensCoefficients()
	{
		m_givensCoeffs.C = Eigen::MatrixXd::Zero(m_dataLength, m_polynomialOrder);
		m_givensCoeffs.S = Eigen::MatrixXd::Zero(m_dataLength, m_polynomialOrder);

		// Compute and store the coefficients of the Givens rotations to compute a QR decomposition of the systemMatrix.
		auto systemMatrix = createSystemMatrix();
		const auto numRows = static_cast<int>(systemMatrix.rows());
		const auto numCols = static_cast<int>(systemMatrix.cols());

		for (int row = 0; row < numRows; row++)
		{
			for (int col = 0; col < std::min(numCols, row); col++)
			{
				// Determine Givens coefficients for eliminating systemMatrix(row,col) with Pivot element systemMatrix(col,col)
				auto rho = std::pow(systemMatrix(col, col), 2) + std::pow(systemMatrix(row, col), 2);
				rho = std::sqrt(rho);
				m_givensCoeffs.C(row, col) = systemMatrix(col, col) / rho;
				m_givensCoeffs.S(row, col) = systemMatrix(row, col) / rho;
				// update the system matrix accordingly, i.e. apply the Givens rotation to the corresponding matrix rows
				eliminateSystemMatrixEntry(systemMatrix, row, col);
			}
		}
	}

	Eigen::MatrixXd PcwPolynomialPartitioning::createSystemMatrix() const
	{
		/* Example for m_polynomialOrder = 3
			systemMatrix = [ 1    1  1
							 4    2  1
							 9    3  1
							 ...
							 n^2  n  1 ]
		*/

		Eigen::MatrixXd systemMatrix = Eigen::MatrixXd::Ones(m_dataLength, m_polynomialOrder);

		for (int j = m_polynomialOrder - 2; j >= 0; j--)
		{
			systemMatrix.col(j) = Eigen::VectorXd::LinSpaced(m_dataLength, 1, m_dataLength).cwiseProduct(systemMatrix.col(j + 1));
		}

		return systemMatrix;
	}

	void PcwPolynomialPartitioning::eliminateSystemMatrixEntry(Eigen::MatrixXd& systemMatrix, int row, int col) const
	{
		if (row <= col)
		{
			// nothing to do
			return;
		}
		const auto c = m_givensCoeffs.C(row, col);
		const auto s = m_givensCoeffs.S(row, col);

		const Eigen::RowVectorXd upperMatRow = systemMatrix.row(col);
		const Eigen::RowVectorXd lowerMatRow = systemMatrix.row(row);

		systemMatrix.row(col) = c * upperMatRow + s * lowerMatRow;
		systemMatrix.row(row) = -s * upperMatRow + c * lowerMatRow;
	}

	void PcwPolynomialPartitioning::fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::MatrixXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const
	{
		const auto leftBound = segment->leftBound;
		const auto& givensRotatedSegmentData = segment->data;
		const auto segmentSize = segment->size();

		// Compute the best fitting polynomial coefficients for the data within the segment
		const Eigen::MatrixXd& rhs = givensRotatedSegmentData.leftCols(m_polynomialOrder).transpose();
		const Eigen::MatrixXd polynomialCoeffs = partialUpperTriMat.topRows(m_polynomialOrder).triangularView<Eigen::Upper>().solve(rhs).transpose();
		// Fill the segment with the induced signal values
		const auto segmentSystemMatrix = Eigen::Map<const Eigen::MatrixXd>(m_fullSystemMatrixTr.leftCols(segmentSize).data(), m_polynomialOrder, segmentSize);
		resultToBeFilled.middleCols(leftBound, segmentSize) = polynomialCoeffs * segmentSystemMatrix;
	}

	std::unique_ptr<ApproxIntervalBase> PcwPolynomialPartitioning::createIntervalForPartitionFinding(const int leftBound, const Eigen::Map<Eigen::MatrixXd>& fullData) const
	{
		return std::make_unique<ApproxIntervalPolynomial>(ApproxIntervalPolynomial(leftBound, fullData, m_polynomialOrder, m_numChannels));
	}

	std::unique_ptr<ApproxIntervalBase> PcwPolynomialPartitioning::createIntervalForComputingResult(const int leftBound, const int rightBound, const Eigen::Map<Eigen::MatrixXd>& fullData) const
	{
		return std::make_unique<ApproxIntervalPolynomial>(ApproxIntervalPolynomial(leftBound, rightBound, fullData, m_polynomialOrder));
	}
}