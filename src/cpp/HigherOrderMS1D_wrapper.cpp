#include <iostream>
#include "mex.h"
#include "HigherOrderMS_1D.h"

                
void mexFunction(int nlhs,  mxArray *plhs[], int nrhs, 
                  const mxArray *prhs[]){
    // Shortcuts for inputs and outputs
    #define J_OUT       plhs[0]
    #define U_OUT       plhs[1]
    
    #define DATA_IN     prhs[0]
    #define K_IN        prhs[1]
    #define GAMMA_IN    prhs[2]
    #define BETA_IN     prhs[3]
    // Catch obvious error
    if (nrhs != 4){
        mexErrMsgTxt("Too few or many input arcloseguments");
    }
    // Get model parameters and pointer to the data
    const double gamma = mxGetScalar(GAMMA_IN);
    const double beta  = mxGetScalar(BETA_IN);
    const int k      = mxGetScalar(K_IN);
    const int n      = mxGetM(DATA_IN);
    double* data_raw = mxGetPr(DATA_IN);
    // Create output
    J_OUT = mxCreateDoubleMatrix(n,1,mxREAL);
    U_OUT = mxCreateDoubleMatrix(n,1,mxREAL);
    double* J_raw = mxGetPr(J_OUT);
    double* u_raw = mxGetPr(U_OUT);
    // Call C++ functions
    armadilloConverter(data_raw,n,k,gamma,beta,J_raw,u_raw);
    
    return;
}
