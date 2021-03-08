addpath(genpath('src'))
clear;
rng(1);
load data_demo.mat
%% Piecewise smooth reconstruction with k-th order Mumford-Shah models 
% model parameters: gamma, beta 
% gamma: complexity parameter(larger gamma -> fewer segments)
% beta: smoothing parameter (larger beta -> stronger smoothing)
sigma = 0.025;
data = pcw_smooth_signal + sigma*randn(size(pcw_smooth_signal));

k     = 1;
gamma = 0.03;
beta  = 2;
[u_1,changePoints_1] = higherOrderMumShah1D(data,gamma,'order',k,'beta',beta);

k     = 2;
gamma = 0.015;
beta  = 2;
[u_2,changePoints_2] = higherOrderMumShah1D(data,gamma,'order',k,'beta',beta);

k     = 3;
gamma = 0.01;
beta  = 2;
[u_3,changePoints_3] = higherOrderMumShah1D(data,gamma,'order',k,'beta',beta);

k     = 4; 
gamma = 0.01;
beta  = 2;
[u_4,changePoints_4] = higherOrderMumShah1D(data,gamma,'order',k,'beta',beta);

% Plot the results
yAxisLim = [0.9*min(data), 1.1*max(data)];
figure('Renderer', 'painters', 'Position', [0 0 1200 600])
subplot(3,2,1)
plotMumShah(pcw_smooth_signal,pcw_smooth_changePoints)
ylim(yAxisLim)
title('Clean piecewise smooth signal')
subplot(3,2,2)
plot(data,'.')
ylim(yAxisLim)
title('Noisy data')
subplot(3,2,3)
plotMumShah(u_1,changePoints_1)
ylim(yAxisLim)
title(['First order Mumford-Shah model'])
subplot(3,2,4)
plotMumShah(u_2,changePoints_2)
ylim(yAxisLim)
title(['Second order Mumford-Shah model'])
subplot(3,2,5)
plotMumShah(u_3,changePoints_3)
ylim(yAxisLim)
title(['Third order Mumford-Shah model'])
subplot(3,2,6)
plotMumShah(u_4,changePoints_4)
ylim(yAxisLim)
title(['Fourth order Mumford-Shah model'])

%% Piecewise polynomial models (k-th order Potts models)
% "Blocks" signal and piecewise constant model (k=1,beta=inf)
sigma = 0.1;
data = blocks_signal + sigma*randn(size(blocks_signal));

k = 1;
gamma = 0.1;
[u,changePoints] = higherOrderMumShah1D(data,gamma,'order',k,'beta',inf);
% Plot the result
yAxisLim = [0.9*min(data), 1.1*max(data)];
figure('Renderer', 'painters', 'Position', [0 0 1200 600])
subplot(1,3,1)
plotMumShah(blocks_signal,blocks_changePoints)
title('Clean "blocks" signal')
ylim(yAxisLim)
subplot(1,3,2)
plot(data,'.')
title('Noisy data')
ylim(yAxisLim)
subplot(1,3,3)
plotMumShah(u,changePoints)
title('Pcw. constant (Potts) reconstruction')
ylim(yAxisLim)

% "Slopes" signal and piecewise affine-linear model (k=2,beta=inf)
sigma = 0.1;
data = slopes_signal + sigma*randn(size(slopes_signal));

k = 2;
gamma = 0.1;
[u,changePoints] = higherOrderMumShah1D(data,gamma,'order',k,'beta',inf);
% Plot the result
yAxisLim = [0.9*min(data), 1.1*max(data)];
figure('Renderer', 'painters', 'Position', [0 0 1200 600])
subplot(1,3,1)
plotMumShah(slopes_signal,slopes_changePoints)
title('Clean "slopes" signal')
ylim(yAxisLim)
subplot(1,3,2)
plot(data,'.')
title('Noisy data')
ylim(yAxisLim)
subplot(1,3,3)
plotMumShah(u,changePoints)
title('Pcw. affine-linear reconstruction')
ylim(yAxisLim)

% "Parabolas" signal and piecewise quadratic model (k=3,beta=inf)
sigma = 0.1;
data = parabolas_signal + sigma*randn(size(parabolas_signal));

k = 3;
gamma = 0.5;
[u,changePoints] = higherOrderMumShah1D(data,gamma,'order',k,'beta',inf);
% Plot the result
yAxisLim = [0.9*min(data), 1.1*max(data)];
figure('Renderer', 'painters', 'Position', [0 0 1200 600])
subplot(1,3,1)
plotMumShah(parabolas_signal,parabolas_changePoints)
title('Clean "parabolas" signal')
ylim(yAxisLim)
subplot(1,3,2)
plot(data,'.')
title('Noisy data')
ylim(yAxisLim)
subplot(1,3,3)
plotMumShah(u,changePoints)
title('Pcw. quadratic reconstruction')
ylim(yAxisLim)
