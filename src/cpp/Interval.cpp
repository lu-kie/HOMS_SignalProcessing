#include "Interval.h"

namespace HOMS
{
	int Interval::getLength() const
	{
		return rightBound - leftBound + 1;
	}


	void Interval::addNewDataPoint(const GivensCoefficients& givensCoeffs, double newDataPoint)
	{
		// Aux variables
		double finiteDifferenceRowData = 0;
		const auto intervalLength = getLength();

		// Apply the Givens rotation which eliminates the new row of the (virtual) system matrix
		// to the interval data to update the approximation error
		for (int j = 0; j <= smoothnessOrder; j++)
		{
			if ((std::isinf(smoothnessPenalty) && (j == smoothnessOrder || j >= intervalLength))
				|| (!std::isinf(smoothnessPenalty) && intervalLength < smoothnessOrder)
				)
			{
				// nothing to do: system matrix either remains upper triangular as it that small or
				// the matrix coefficient must not be eliminated as it is the upper right quadrant
				break;
			}

			double pivotRowData;
			if (j != smoothnessOrder)
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

			if (j < smoothnessOrder)
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
			approxError += (intervalLength > smoothnessOrder - 1) ? std::pow(newDataPoint, 2) : 0;
		}
		else
		{
			approxError += (intervalLength > smoothnessOrder - 1) ? std::pow(finiteDifferenceRowData, 2) : 0;
		}

		// Update the interval length
		rightBound++;

		// Update the stored interval data if necessary
		if (std::isinf(smoothnessPenalty))
		{
			if (intervalLength < smoothnessOrder)
			{
				data(intervalLength) = newDataPoint;
			}
		}
		else
		{
			if (smoothnessOrder > 1)
			{
				data.head(smoothnessOrder - 1) = data.segment(1, smoothnessOrder - 1);
			}
			data(smoothnessOrder - 1) = newDataPoint;
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

		const double pivotRowData = data(col);
		const auto intervalLength = getLength();
		const double eliminatedRowData = data(row);//std::isinf(smoothnessPenalty) ? data(row) : data(row + intervalLength);
		// Apply Givens rotation to data vector
		double c, s;
		if (std::isinf(smoothnessPenalty))
		{
			c = givensCoeffs.C(row, col);
			s = givensCoeffs.S(row, col);
		}
		else
		{
			c = givensCoeffs.C(row - rowOffset, col - colOffset);
			s = givensCoeffs.S(row - rowOffset, col - colOffset);
		}
		data(col) = c * pivotRowData + s * eliminatedRowData;
		data(row) = -s * pivotRowData + c * eliminatedRowData;
	}
}