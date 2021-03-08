#include "Interval.h"

// Function definitions
Interval::~Interval()
{
}
// Constructur for usage in findBestPartition
Interval::Interval(int left,int right, double data_new, const int k, double alpha)
{
	beta = alpha;
    l = left;
    r = right;
	data = zeros(k);
	if (std::isinf(beta)){
		data(0) = data_new;
	} else{
		data(k-1) = data_new;
	}
    eps = 0.0;
}

// Constructor for usage in reconstructionFromPartition
Interval::Interval(int left, int right, vec y, const int k, double alpha)
{
	beta = alpha;
    l = left;
    r = right;
	int h = giveLength();
	if (std::isinf(beta)){
		data = y;
	} else{
		// For the reconstruction process the data vector y must be appended by zeros for piecewise smooth reconstruction
		data = zeros(2*h-k);
		data.rows(0,h-1) = y;
	}
    eps = 0.0;
}

// Getter
int Interval::getL()
{
    return l;
}

int Interval::getR()
{
    return r;
}

double Interval::getEps()
{
    return eps;
}

double Interval::getBeta()
{
	return beta;
}

vec Interval::getData()
{
    return data;
}

// Setter
void Interval::setEps(double e)
{
    eps = e;
}
void Interval::setL(int left)
{
    l = left;
}
void Interval::setR(int right)
{
    r = right;
}
void Interval::setData(vec y)
{
    data = y;
}
// Give interval / data length
int Interval::giveLength()
{
    return r-l+1;
}

// Add new data point to the interval and update the approximation error
void Interval::addBottomDataPoint(const int k, mat &C, mat &S, double data_new)
{
	// Aux variables
    double c,s,f_j,f_r,data_diff=0;
    int h = giveLength();
    
	// Eliminate the new row
	for(int j = 0; j <= k; j++) {
		if (std::isinf(beta) && (j == k || j >= h)){
			break;
		}
		if (!std::isinf(beta) && h < k){
			break;
		}
		c = C(h,j);
		s = S(h,j);
		f_j = (j != k) ? data(j) : data_new;
		f_r = std::isinf(beta) ? data_new : data_diff;
		// Givens transform data
		if (j < k) {
			data(j)  = c*f_j + s*f_r;
		}
		else{
			data_new = c*f_j + s*f_r;
		}
		if (std::isinf(beta)){
			data_new  = -s*f_j + c*f_r;
		} else{
			data_diff = -s*f_j + c*f_r;
		}
	}
	// Update interval approximation error
	if (std::isinf(beta)){
		eps += (h > k-1) ? (data_new*data_new) : 0;
	} else{
		eps += (h > k-1) ? (data_diff*data_diff) : 0;
	}
	// Update the interval length
    r++;
    // Update interval data
	if(std::isinf(beta)){
		if (h < k)
			data(h) = data_new;
	} else{
		if (k > 1){
			data.rows(0,k-2) = data.rows(1,k-1);		
		}
		data(k-1) = data_new;
	}
}

// Givens rotate data for reconstructionFromPartition
void Interval::givensRotate(double c, double s,int v, int vv)
{
	// Aux variables
    double f_j = data(v);
	int h = giveLength();
    double f_i = std::isinf(beta) ? data(vv) : data(vv+h);
    // Apply Givens rotation to data vector
    data(v) = c*f_j+s*f_i;
	if (std::isinf(beta)){
		data(vv) = -s*f_j+c*f_i;
	} else{
		data(vv+h) = -s*f_j+c*f_i;
	}
}
