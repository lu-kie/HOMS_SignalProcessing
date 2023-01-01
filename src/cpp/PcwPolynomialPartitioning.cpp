#include "PcwPolynomialPartitioning.h"
#include "Interval.h"

namespace HOMS
{
	int PcwPolynomialPartitioning::minSegmentSize() const
	{
		return m_polynomialOrder + 1;
	}

	GivensCoefficients PcwPolynomialPartitioning::createGivensCoefficients() const
	{
		return GivensCoefficients(m_dataLength, m_polynomialOrder, std::numeric_limits<double>::infinity(), computeSystemMatrix());
	}

	Eigen::MatrixXd PcwPolynomialPartitioning::computeSystemMatrix() const
	{
		/* Example for k = 3
			A = [ 1    1  1
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

	void PcwPolynomialPartitioning::fillSegmentFromPartialUpperTriangularSystemMatrix(IntervalBase* segment, Eigen::VectorXd& resultToBeFilled, const Eigen::MatrixXd& partialUpperTriMat) const
	{
		const auto leftBound = segment->leftBound;
		const auto rightBound = segment->rightBound;
		const auto& currData = segment->data;
		const auto segmentSize = segment->size();
		const Eigen::Index numRows = partialUpperTriMat.cols(); // = polynomial order
		// Compute segment's polynomial coefficients
		const Eigen::VectorXd polynomialCoeffs = partialUpperTriMat.topRows(numRows).triangularView<Eigen::Upper>().solve(currData.head(numRows));

		// Fill the segment with values induced by the polynomial coefficients
		resultToBeFilled.segment(leftBound, segmentSize).array() += polynomialCoeffs.tail(1).value();
		Eigen::VectorXd xValues = Eigen::VectorXd::LinSpaced(segmentSize, 1, segmentSize);
		for (int j = static_cast<int>(polynomialCoeffs.size()) - 2; j >= 0; j--)
		{
			resultToBeFilled.segment(leftBound, segmentSize) += polynomialCoeffs(j) * xValues;
			xValues.array() *= xValues.array();
		}
	}


	std::unique_ptr<IntervalBase> PcwPolynomialPartitioning::createIntervalForPartitionFinding(const int leftBound, const double newDataPoint) const
	{
		return std::make_unique<IntervalPolynomial>(IntervalPolynomial(leftBound, newDataPoint, m_polynomialOrder));
	}

	std::unique_ptr<IntervalBase> PcwPolynomialPartitioning::createIntervalForComputingPcwSmoothSignal(const int leftBound, const int rightBound, const Eigen::VectorXd& data) const
	{
		return std::make_unique<IntervalPolynomial>(IntervalPolynomial(leftBound, rightBound, data, m_polynomialOrder));
	}
}