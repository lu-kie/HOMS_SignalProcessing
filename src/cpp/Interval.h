#ifndef INTERVAL_H
#define INTERVAL_H

#define ARMA_NO_DEBUG
#include <armadillo>

using namespace arma;

class Interval
{
private:
    int l; //Left bound of discrete interval
    int r; //Right bound of discrete interval
    double eps; //Approximation error for interval
	double beta; // Smoothing penalty of interval
    vec data; // Interval data
public:
    // Getter:
    int getL();
    int getR();
    double getEps();
	double getBeta();
    vec getData();
    // Setter:
    void setEps(double e);
    void setL(int left);
    void setR(int right);
    void setData(vec data);
    // Destructor
    ~Interval();
	// Constructor for finding an optimal partition
    Interval(int left, int right, double data_new, const int k, double alpha);
    // Constructor for the signal reconstruction from a partition
    Interval(int left, int right, vec y, const int k, double alpha);
    // Give interval / data length
    int giveLength() ;
    // Add data point to the bottom of the interval for updating the corresp. approximation error
    void addBottomDataPoint(const int k, mat &C, mat &S, double data_new);
    // Update associated data i.e. sparse Givens rotate it (for reconstruction process)
    void givensRotate(double c, double s,int t, int w);
};

#endif