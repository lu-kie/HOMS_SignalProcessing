# HOMS_SignalProcessing
HOMS_SignalProcessing is a C++ toolbox with MATLAB wrapper for joint smoothing and partitioning of one-dimensional signals and time series 
using higher order Mumford-Shah and Potts models.

   - Fast and numerically stable algorithms for solving univariate Mumford-Shah problems of any order
   - The used dynamic programming scheme computes a global minimum
   - Theoretical quadratic runtime guarantee
   - Practical linear runtime by using pruning strategies

This is an implementation of the algorithms described in the paper

>  **M. Storath, L. Kiefer, A. Weinmann**
    *Smoothing for signals with discontinuities using higher order Mumford-Shah models.*
    Numerische Mathematik, 2019.

## Joint Smoothing and Partitioning of 1D Data and Time Series
### Piecewise smooth models
   
   Recovering a piecewise smooth signal from noisy data:
   classical spline approximation smooths out the discontinuities;
   the classical first order Mumford–Shah model allows for discontinuities, but the estimate misses most of them;
   higher order Mumford–Shah models provide improved smoothing and
   segmentation.
   
   ![titleimageA](/docs/pcwSmooth.png)

   
 ### Piecewise polynomial models
  Recovering a piecewise constant signal from noisy data with the piecewise constant Mumford-Shah model (aka Potts model).
  ![titleimageB](/docs/pcwConstant.png)
  
  Recovering a piecewise affine-linear signal from noisy data with the piecewise affine-linear Mumford-Shah model.
  ![titleimageC](/docs/pcwLinear.png)
  
  
  Recovering a piecewise quadratic signal from noisy data with the piecewise quadratic Mumford-Shah model.
  ![titleimageD](/docs/pcwQuadratic.png)
  

## Installation
### Compiling
The algorithm depends on a mex script that needs to be compiled before execution. For compilation inside MATLAB, cd into the 'src/cpp' folder and run build.m

Requires the Armadillo library

Tested with Armadillo 8.400 https://launchpad.net/ubuntu/+source/armadillo/1:8.400.0+dfsg-2

On Linux, just use your package manager to install it:

sudo apt-get install libarmadillo-dev


### Running
For a test run, run demo.m. 
demo.m calls the wrapper function higherOrderMumShah1D.m which calls the necessary C++ sources

Arguments of higherOrderMumShah1D are:
 - data: input data array (double)
 - gamma: complexity penalty (larger choice -> fewer segments)
 - beta: smoothing parameter (larger choice -> stronger smoothing; beta=inf -> piecewise polynomial model)
 - order: order of the applied Mumford-Shah or Potts model (polynomial trends up to the specified order will be preserved)
 - varargin: optional input parameters

## References
- M. Storath, L. Kiefer, A. Weinmann
    "Smoothing for signals with discontinuities using higher order Mumford-Shah models."
    Numerische Mathematik, 2019.
