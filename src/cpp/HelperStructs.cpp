#include "HelperStructs.h"

namespace homs
{
	Partitioning::Partitioning(const std::vector<int>& jumpsTracker)
	{
		segments.reserve(jumpsTracker.size());
		auto rightBound = static_cast<int>(jumpsTracker.size() - 1);
		while (true)
		{
			const auto leftBound = static_cast<int>(jumpsTracker.at(rightBound)) + 1;

			segments.push_back(Segment(leftBound, rightBound));
			if (leftBound == 0)
			{
				break;
			}
			rightBound = leftBound - 1;
		}
		std::reverse(segments.begin(), segments.end());
	}

	void ApproxIntervalPolynomial::addNewDataPoint(const GivensCoefficients& givensCoeffs, Eigen::VectorXd&& newDataPoint)
	{
		// Aux variables
		double finiteDifferenceRowData = 0;
		const auto intervalLength = size();

		// Apply the Givens rotation which eliminates the new row of the (virtual) system matrix
		// to the interval data to update the approximation error
		for (int j = 0; j < std::min(polynomialOrder, intervalLength); j++)
		{
			// Apply the Givens transform to the data
			const auto c = givensCoeffs.C(intervalLength, j);
			const auto s = givensCoeffs.S(intervalLength, j);

			const Eigen::VectorXd pivotRowData = data.col(j);
			const Eigen::VectorXd eliminatedRowData = newDataPoint;
			data.col(j) = c * pivotRowData + s * eliminatedRowData;
			newDataPoint = -s * pivotRowData + c * eliminatedRowData;
		}

		const auto newIntervalLength = intervalLength + 1;

		// Update the approximation error
		if (newIntervalLength > polynomialOrder)
		{
			approxError += newDataPoint.squaredNorm();
		}

		// Update the interval boundaries
		rightBound++;

		// Update the stored interval data if necessary
		if (newIntervalLength <= polynomialOrder)
		{
			data.col(newIntervalLength - 1) = newDataPoint;
		}
	}

	void ApproxIntervalPolynomial::applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col)
	{
		if (col >= row)
		{
			// nothing to be eliminated in system matrix: do nothing
			return;
		}

		// Apply Givens rotation to data vector
		const auto c = givensCoeffs.C(row, col);
		const auto s = givensCoeffs.S(row, col);

		const Eigen::VectorXd pivotRowData = data.col(col);
		const Eigen::VectorXd eliminatedRowData = data.col(row);
		data.col(col) = c * pivotRowData + s * eliminatedRowData;
		data.col(row) = -s * pivotRowData + c * eliminatedRowData;
	}

	void ApproxIntervalSmooth::addNewDataPoint(const GivensCoefficients& givensCoeffs, Eigen::VectorXd&& newDataPoint)
	{
		// Aux variables
		Eigen::VectorXd finiteDifferenceRowData = Eigen::VectorXd::Zero(newDataPoint.rows());
		const auto intervalLength = size();

		// Apply the Givens rotation which eliminates the new row of the (virtual) system matrix
		// to the interval data to update the approximation error
		if (intervalLength >= smoothingOrder)
		{
			for (int j = 0; j <= smoothingOrder; j++)
			{
				Eigen::VectorXd pivotRowData;
				if (j != smoothingOrder)
				{
					pivotRowData = data.col(j);
				}
				else
				{
					pivotRowData = newDataPoint;
				}

				auto eliminatedRowData = finiteDifferenceRowData;

				// Apply the Givens transform to the data
				const auto c = givensCoeffs.C(intervalLength, j);
				const auto s = givensCoeffs.S(intervalLength, j);

				if (j != smoothingOrder)
				{
					data.col(j) = c * pivotRowData + s * eliminatedRowData;
				}
				else
				{
					newDataPoint = c * pivotRowData + s * eliminatedRowData;
				}

				finiteDifferenceRowData = -s * pivotRowData + c * eliminatedRowData;
			}
			// Update the approximation error
			approxError += finiteDifferenceRowData.squaredNorm();
		}

		// Update the interval length
		rightBound++;

		// Update the stored interval data if necessary
		if (smoothingOrder > 1)
		{
			data.leftCols(smoothingOrder - 1) = data.middleCols(1, smoothingOrder - 1);
		}
		data.col(smoothingOrder - 1) = newDataPoint;
	}

	void ApproxIntervalSmooth::applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col)
	{
		const auto fullSignalLength = static_cast<int>(givensCoeffs.C.rows());
		if (row < fullSignalLength)
		{
			return;
		}
		const auto rowOffset = fullSignalLength - smoothingOrder;
		const auto eliminatedRowDataIndexOffset = fullSignalLength - size();
		const auto colOffset = row - fullSignalLength;

		const Eigen::VectorXd pivotRowData = data.col(col + colOffset);
		const Eigen::VectorXd eliminatedRowData = data.col(row - eliminatedRowDataIndexOffset);

		// Apply Givens rotation to data vector
		const auto c = givensCoeffs.C(row - rowOffset, col);
		const auto s = givensCoeffs.S(row - rowOffset, col);
		data.col(col + colOffset) = c * pivotRowData + s * eliminatedRowData;
		data.col(row - eliminatedRowDataIndexOffset) = -s * pivotRowData + c * eliminatedRowData;
	}
}