function [u,changePoints] = higherOrderMumShah1D(data,gamma,varargin)
% [u,changePoints] = higherOrderMumShah1D(f,gamma,parameters)
% Computes a minimizer of Mumford-Shah models of any order with the
% dynamic programming approach of [1].
%
% Inputs:
%   data: vector with the data to be partitioned and piecewise smoothed
%   gamma: complexity parameter which determines the cost of introducing a
%   changepoint; lower values lead to more segments
%   'order': positive integer that determines the order of the model
%   (polynomial trends up to the order are free and will be preserved)
%   'beta': smoothing parameter which determines how much the data are
%   smoothed; higher values lead to stronger smoothing
%
% Outputs:
%   u: piecewise smoothed signal
%   changePoints: segment boundaries of the corresponding optimal partitioning
%
% Ref:
% [1] Storath,Kiefer,Weinmann - Smoothing for signals with discontinuities
%     using higher order Mumford–Shah models, Numerische Mathematik
%
%
% Copyright (c) 2021 Lukas Kiefer <lukas.kiefer2@gmail.com>
%
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU Affero General Public License as published by
% the Free Software Foundation, either version 3 of the License,
% or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
% License for more details.
%
% You should have received a copy of the GNU Affero General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
%%
% Parse options
ip = inputParser;
% If no order and smoothing penalty are specified, the classical piecewise constant 
% Mumford-Shah model ("Potts model") is applied
addParameter(ip,'order', 1);
addParameter(ip,'beta', inf);
parse(ip, varargin{:});
par = ip.Results;
% Check input arguments
assert(gamma > 0, 'Complexity parameter gamma must be > 0.');
assert(par.beta  > 0, 'Smoothing parameter beta must be > 0.');
% The solver expects a column vector
assert(min(size(data)) == 1, 'Data must be one-dimensional.');
row_vector = false;
if size(data,2) > size(data,1)
    data = data';
    row_vector = true;
end
% Call the mex wrapper of the C++ solver
[J,u] = HigherOrderMS1D_wrapper(data,par.order,gamma,par.beta);
% Get the changePoints encoded by J
changePoints = getChangePoints(J);
% Undo transposition if necessary
if row_vector
    u = u';
end
end

function changePoints = getChangePoints(J)
n = length(J);
r = n;
i = 1;
changePoints = [];
while(true)
    l = J(r)+1;
    if l == 1
        break;
    end
    changePoints(i,:) = [l-1 l];
    r = l-1;
    i = i+1;
end
changePoints = flipud(changePoints);
end