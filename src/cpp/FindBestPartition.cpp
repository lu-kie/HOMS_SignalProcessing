/**
    FindBestPartition.cpp
    Purpose: Computes a best partition in terms of the one-dimensional 
             k-th order Mumford-Shah model for univariate data
             for jump penalty gamma and smoothing penalty beta by dynamic programming

    @author Lukas Kiefer
    @version 1.0
*/

#include "HigherOrderMS_1D.h"
void findBestPartition(vec &data,const int n, const double gamma, vec &eps_1r,
                          const int k, const double beta, mat &C, mat &S,vec &J)
{
    // Allocate vector with optimal functional values for each r=1,...,n
    vec B = zeros(n,1);
    J(0) = 0;
    // List which will contain the candidate segments, i.e., discrete intervals
    std::list<Interval> segments;
    segments.push_front(Interval(2,2,data(1),k, beta));
    // Declare aux variables
    double b,prun_I,prun_II,data_new,eps_curr;
    int l_curr,r_curr;
    bool seg_deleted;
	
    for(int r = 2; r <= n; r++){
        // Init with approximation error of single-segment partition, i.e. l=1
        B(r-1) = eps_1r(r-1);
        J(r-1) = 0;
        // Loop (backwards in l) through candidates for the best last jump for data(1:r)
        for(list<Interval>::iterator it = segments.begin(); it != segments.end();){
            seg_deleted = false;
            Interval &curr_interval = *it;
            while (curr_interval.getR() < r){
                l_curr = curr_interval.getL();
                r_curr = curr_interval.getR();
                eps_curr = curr_interval.getEps();
                // Pruning strategy I (discard potential segments/intervals which can never be part of an optimal partitioning)
                prun_I = B(l_curr-2)+eps_curr;
                if (r>2 && r_curr < r  &&   prun_I >= B(r_curr-1)){
                    // Delete current interval from list as it will never be optimal
                    it = segments.erase(it);
                    seg_deleted = true;
                    break;
                } else{
                    // Extend current interval by new data and update its approximation error with Givens rotations
                    data_new = data(r_curr);
                    curr_interval.addBottomDataPoint(k,C,S,data_new);
                }
            }
            if (seg_deleted){
                continue;
            }
            l_curr = curr_interval.getL();
            r_curr = curr_interval.getR();
            eps_curr = curr_interval.getEps();
            // Check if the current interval yields an improved energy value
            b = B(l_curr - 2) + gamma + eps_curr;
            if (b <= B(r-1)) {
                B(r-1) = b;
                J(r-1) = l_curr-1;
            }
            // Pruning strategy II (omit unnecessary computations of approximation errors)
            prun_II = eps_curr+gamma;
            if (prun_II > B(r-1)){
                break;
            }
            ++it;
        }
        // Add the interval with left bound = r to the list of candidates
        if (r <= n-1){
            segments.push_front(Interval(r+1,r+1,data(r),k, beta));
        }
    }
}
