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
				if (systemMatrix(col, col) < 0)
				{
					rho = -rho;
				}
				m_givensCoeffs.C(row, col) = systemMatrix(col, col) / rho;
				m_givensCoeffs.S(row, col) = systemMatrix(row, col) / rho;
				// update the system matrix accordingly, i.e. apply the Givens rotation to the corresponding matrix rows
				Eigen::MatrixXd upperMatRow = systemMatrix.row(col);
				Eigen::MatrixXd lowerMatRow = systemMatrix.row(row);
				systemMatrix.row(col) = m_givensCoeffs.C(row, col) * upperMatRow + m_givensCoeffs.S(row, col) * lowerMatRow;
				systemMatrix.row(row) = -m_givensCoeffs.S(row, col) * upperMatRow + m_givensCoeffs.C(row, col) * lowerMatRow;
			}
		}
	}

	Eigen::MatrixXd PcwPolynomialPartitioning::createSystemMatrix() const
	{
		/* Example for m_smoothingOrder = 3
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

		const Eigen::VectorXd upperMatRow = systemMatrix.row(col);
		const Eigen::VectorXd lowerMatRow = systemMatrix.row(row);

		systemMatrix.row(col) = c * upperMatRow + s * lowerMatRow;
		systemMatrix.row(row) = -s * upperMatRow + c * lowerMatRow;
	}

	void PcwPolynomialPartitioning::fillSegmentFromPartialUpperTriangularSystemMatrix(ApproxIntervalBase* segment, Eigen::VectorXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const
	{
		const auto leftBound = segment->leftBound;
		const auto rightBound = segment->rightBound;
		const auto& currData = segment->data;
		const auto segmentSize = segment->size();
		const Eigen::Index numRows = partialUpperTriMat.cols(); // = polynomial order
		
		// Compute the best fitting polynomial coefficients for the data within the segment
		const Eigen::VectorXd polynomialCoeffs = partialUpperTriMat.topRows(numRows).triangularView<Eigen::Upper>().solve(currData.head(numRows));

		// Fill the segment with the values yielded by the polynomial coefficients
		resultToBeFilled.segment(leftBound, segmentSize).array() += polynomialCoeffs.tail(1).value();
		Eigen::VectorXd xValues = Eigen::VectorXd::LinSpaced(segmentSize, 1, segmentSize);
		for (int j = static_cast<int>(polynomialCoeffs.size()) - 2; j >= 0; j--)
		{
			resultToBeFilled.segment(leftBound, segmentSize) += polynomialCoeffs(j) * xValues;
			xValues.array() *= xValues.array();
		}
	}

	std::unique_ptr<ApproxIntervalBase> PcwPolynomialPartitioning::createIntervalForPartitionFinding(const int leftBound, const double newDataPoint) const
	{
		return std::make_unique<ApproxIntervalPolynomial>(ApproxIntervalPolynomial(leftBound, newDataPoint, m_polynomialOrder));
	}

	std::unique_ptr<ApproxIntervalBase> PcwPolynomialPartitioning::createIntervalForComputingResult(const int leftBound, const int rightBound, const Eigen::VectorXd& data) const
	{
		return std::make_unique<ApproxIntervalPolynomial>(ApproxIntervalPolynomial(leftBound, rightBound, data, m_polynomialOrder));
	}
}