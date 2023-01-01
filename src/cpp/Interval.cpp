#include "Interval.h"

namespace HOMS
{
	int Interval::size() const
	{
		return rightBound - leftBound + 1;
	}


	void Interval::addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint)
	{
		// Aux variables
		double finiteDifferenceRowData = 0;
		const auto intervalLength = size();

		// Apply the Givens rotation which eliminates the new row of the (virtual) system matrix
		// to the interval data to update the approximation error
		for (int j = 0; j <= smoothingOrder; j++)
		{
			if ((std::isinf(smoothnessPenalty) && (j == smoothingOrder || j >= intervalLength))
				|| (!std::isinf(smoothnessPenalty) && intervalLength < smoothingOrder)
				)
			{
				// nothing to do: system matrix either remains upper triangular as it that small or
				// the matrix coefficient must not be eliminated as it is the upper right quadrant
				break;
			}

			double pivotRowData;
			if (j != smoothingOrder)
			{
				pivotRowData = data(j);
			}
			else
			{
				pivotRowData = newDataPoint;
			}

			double eliminatedRowData;
			if (std::isinf(smoothnessPenalty))
			{
				eliminatedRowData = newDataPoint;
			}
			else
			{
				eliminatedRowData = finiteDifferenceRowData;
			}

			// Apply the Givens transform to the data
			const auto c = givensCoeffs.C(intervalLength, j);
			const auto s = givensCoeffs.S(intervalLength, j);

			if (j < smoothingOrder)
			{
				data(j) = c * pivotRowData + s * eliminatedRowData;
			}
			else
			{
				newDataPoint = c * pivotRowData + s * eliminatedRowData;
			}

			if (std::isinf(smoothnessPenalty))
			{
				newDataPoint = -s * pivotRowData + c * eliminatedRowData;
			}
			else
			{
				finiteDifferenceRowData = -s * pivotRowData + c * eliminatedRowData;
			}
		}
		// Update the approximation error
		if (std::isinf(smoothnessPenalty))
		{
			approxError += (intervalLength > smoothingOrder - 1) ? std::pow(newDataPoint, 2) : 0;
		}
		else
		{
			approxError += (intervalLength > smoothingOrder - 1) ? std::pow(finiteDifferenceRowData, 2) : 0;
		}

		// Update the interval length
		rightBound++;

		// Update the stored interval data if necessary
		if (std::isinf(smoothnessPenalty))
		{
			if (intervalLength < smoothingOrder)
			{
				data(intervalLength) = newDataPoint;
			}
		}
		else
		{
			if (smoothingOrder > 1)
			{
				data.head(smoothingOrder - 1) = data.segment(1, smoothingOrder - 1);
			}
			data(smoothingOrder - 1) = newDataPoint;
		}
	}

	void Interval::applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col)
	{
		Interval::applyGivensRotationToData(givensCoeffs, row, col, 0, 0);
	}

	void Interval::applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col, const int rowOffset, const int colOffset)
	{
		if (std::isinf(smoothnessPenalty) && col >= row)
		{
			// nothing to be eliminated in system matrix: do nothing
			return;
		}

		int eliminatedRowDataIndexOffset = 0;
		if (!std::isinf(smoothnessPenalty))
		{
			eliminatedRowDataIndexOffset = static_cast<int>(givensCoeffs.C.rows()) - size();
		}

		const double pivotRowData = data(col + colOffset);
		const double eliminatedRowData = data(row - eliminatedRowDataIndexOffset);

		// Apply Givens rotation to data vector
		double c, s;
		if (std::isinf(smoothnessPenalty))
		{
			c = givensCoeffs.C(row, col);
			s = givensCoeffs.S(row, col);
		}
		else
		{
			c = givensCoeffs.C(row - rowOffset, col);
			s = givensCoeffs.S(row - rowOffset, col);
		}
		data(col + colOffset) = c * pivotRowData + s * eliminatedRowData;
		data(row - eliminatedRowDataIndexOffset) = -s * pivotRowData + c * eliminatedRowData;
	}


	void IntervalPolynomial::addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint)
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

			const auto pivotRowData = data(j);
			const auto eliminatedRowData = newDataPoint;
			data(j) = c * pivotRowData + s * eliminatedRowData;
			newDataPoint = -s * pivotRowData + c * eliminatedRowData;

		}

		const auto newIntervalLength = intervalLength + 1;

		// Update the approximation error
		if (newIntervalLength > polynomialOrder)
		{
			approxError += std::pow(newDataPoint, 2);
		}

		// Update the interval boundaries
		rightBound++;

		// Update the stored interval data if necessary
		if (newIntervalLength <= polynomialOrder)
		{
			data(newIntervalLength - 1) = newDataPoint;
		}
	}

	void IntervalPolynomial::applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col)
	{
		if (col >= row)
		{
			// nothing to be eliminated in system matrix: do nothing
			return;
		}

		// Apply Givens rotation to data vector
		const auto c = givensCoeffs.C(row, col);
		const auto s = givensCoeffs.S(row, col);

		const auto pivotRowData = data(col);
		const auto eliminatedRowData = data(row);
		data(col) = c * pivotRowData + s * eliminatedRowData;
		data(row) = -s * pivotRowData + c * eliminatedRowData;
	}


	void IntervalSmooth::addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint)
	{
		// Aux variables
		double finiteDifferenceRowData = 0;
		const auto intervalLength = size();

		// Apply the Givens rotation which eliminates the new row of the (virtual) system matrix
		// to the interval data to update the approximation error
		for (int j = 0; j <= smoothingOrder; j++)
		{
			if (intervalLength < smoothingOrder)
			{
				// nothing to do: system matrix either remains upper triangular as it that small or
				// the matrix coefficient must not be eliminated as it is the upper right quadrant
				break;
			}

			double pivotRowData;
			if (j != smoothingOrder)
			{
				pivotRowData = data(j);
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
				data(j) = c * pivotRowData + s * eliminatedRowData;
			}
			else
			{
				newDataPoint = c * pivotRowData + s * eliminatedRowData;
			}

			finiteDifferenceRowData = -s * pivotRowData + c * eliminatedRowData;
		}
		// Update the approximation error
		approxError += (intervalLength >= smoothingOrder) ? std::pow(finiteDifferenceRowData, 2) : 0;


		// Update the interval length
		rightBound++;

		// Update the stored interval data if necessary
		if (smoothingOrder > 1)
		{
			data.head(smoothingOrder - 1) = data.segment(1, smoothingOrder - 1);
		}
		data(smoothingOrder - 1) = newDataPoint;
	}

	void IntervalSmooth::applyGivensRotationToData(const GivensCoefficients& givensCoeffs, const int row, const int col)
	{
		const auto fullSignalLength = static_cast<int>(givensCoeffs.C.rows());
		if (row < fullSignalLength)
		{
			return;
		}
		const auto rowOffset = fullSignalLength - smoothingOrder;
		const auto eliminatedRowDataIndexOffset = fullSignalLength - size();
		const auto colOffset = row - fullSignalLength;

		const double pivotRowData = data(col + colOffset);
		const double eliminatedRowData = data(row - eliminatedRowDataIndexOffset);

		// Apply Givens rotation to data vector
		const auto c = givensCoeffs.C(row - rowOffset, col);
		const auto s = givensCoeffs.S(row - rowOffset, col);
		data(col + colOffset) = c * pivotRowData + s * eliminatedRowData;
		data(row - eliminatedRowDataIndexOffset) = -s * pivotRowData + c * eliminatedRowData;
	}
}