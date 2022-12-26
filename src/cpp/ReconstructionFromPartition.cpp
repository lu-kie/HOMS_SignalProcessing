/**
	ReconstructionFromPartition.cpp
	Purpose: Computes the corresponding 1D-reconstruction of the input 1D-partition

	@author Lukas Kiefer
	@version 1.0
*/

#include "HigherOrderMS_1D.h"

void reconstructionFromPartition(vec& J, vec& u, vec& data, const int n, const int k, const double beta, mat& C, mat& S)
{
	// Create interval objects corresponding to the optimal partition encoded by J
	std::list<Interval> Intervals;
	int r = n, l;
	while (true) {
		l = J(r - 1) + 1;
		// Handle/Catch intervals with length <= k (smoothed signal equals data)
		if (r - l + 1 <= k) {
			u.rows(l - 1, r - 1) = data.rows(l - 1, r - 1);
		}
		else {
			Intervals.push_front(Interval(l, r, data.rows(l - 1, r - 1), k, beta));
		}
		if (l == 1) {
			break;
		}
		r = l - 1;
	}
	// If all intervals/segments have length <= k, no reconstruction is needed
	if (Intervals.empty()) {
		return;
	}
	// Sort Intervals in length-ascending order
	// (This allows to avoid computing the same row transformations of the system matrix A multiple times.)
	Intervals.sort(compare_intervLengths);
	// Solve the linear equation systems corresponding to each interval
	// Declare sparse system matrix of underlying least squares problem
	mat A = constructSystemMatrix(n, k, beta);
	// Flag for Potts case (polynomials on segments)
	bool potts = isinf(beta);
	// Declare aux variables
	double c, s;
	int v, vv, tt, q, w, ww, off = 0; // Offset variables for the sparse matrices
	int h; // Stores the interval length
	// The "master" iterator knowing which intervals have to be considered (i.e. which interval lengths)
	list<Interval>::iterator master_it = Intervals.begin();
	// Matrix elimination (i: row, j: column)
	for (int i = 0; i < 2 * n - k; i++) {
		if ((!potts && i < n) || (potts && i >= n)) {
			continue;
		}
		for (int j = 0; j < k + 1; j++) {
			if (potts && (j == k || i <= j)) {
				break;
			}
			if (potts) {
				q = i;
				v = j;
				vv = i;
				tt = k - 1;
				w = 0;
				ww = k - 1;
			}
			else {
				q = i - n + k;
				v = j + off;
				vv = i - n;
				tt = k - j;
				w = j;
				ww = k;
			}
			// Get Givens coefficients to eliminate A(i,j)
			c = C(q, j);
			s = S(q, j);
			// Update A (incorporating its sparse declaration)
			rowvec A_j = A.submat(v, 0, v, tt);
			rowvec A_r = A.submat(i, w, i, ww);
			A.submat(v, 0, v, tt) = c * A_j + s * A_r;
			A.submat(i, w, i, ww) = -s * A_j + c * A_r;
			// Update interval data accordingly
			for (list<Interval>::iterator it = master_it; it != Intervals.end(); ++it) {
				Interval& curr_interval = *it;
				curr_interval.givensRotate(c, s, v, vv);
			}
		}
		if (!potts) {
			off++; // update offset aux variable
		}
		// Fill u on finished intervals and delete them from the list of intervals
		while (true) {
			Interval& curr_interval = *master_it;
			h = curr_interval.giveLength();
			// If no interval has the current length, break:
			if ((potts && (h != i + 1)) || (!potts && (h != i - n + k + 1))) {
				break;
			}
			l = curr_interval.getL();
			r = curr_interval.getR();
			vec data_curr = curr_interval.getData();
			if (!potts) {
				// Fill segment via back substitution
				u(r - 1) = data_curr(h - 1) / A(h - 1, 0);
				for (int ii = h - 2; ii >= 0; ii--) {
					double p = 0;
					for (int j = 1; j <= min(k, h - ii - 1); j++) {
						p += A(ii, j) * u(l + ii + j - 1);
					}
					u(l - 1 + ii) = (data_curr(ii) - p) / A(ii, 0);
				}
			}
			else {
				// Compute segment's polynomial coefficients p via back substitution
				vec p = zeros(k);
				p(k - 1) = data_curr(k - 1) / A(k - 1, k - 1);
				for (int ii = k - 2; ii >= 0; ii--) {
					double p_sum = 0;
					for (int j = ii; j < k; j++) {
						p_sum += A(ii, j) * p(j);
					}
					p(ii) = (data_curr(ii) - p_sum) / A(ii, ii);
				}
				// Fill the segment with values induced by p
				for (int j = k; j > 0; j--) {
					u.rows(l - 1, r - 1) = u.rows(l - 1, r - 1) + p(k - j) * pow(linspace<vec>(1, h, h), j - 1);
				}
			}
			// Increase master Iterator (a filled segment won't be considered again)
			++master_it;
			// Check if master iterator has finished, i.e., if reconstructions on all intervals are finished
			if (master_it == Intervals.end()) {
				return;
			}
		}
	}
}


// Aux function for comparing interval lengths for sorting them
bool compare_intervLengths(Interval inter1, Interval inter2)
{
	return (inter1.giveLength() < inter2.giveLength());
}
