function [] = plotMumShah(u,changePoints)
n = length(u);
I = changePoints';
I = [1; I(:); n];

% Piecewise plot of u with respect to the changePoints
hold on
for j = 1:2:length(I)-1
    ax = gca;
    ax.ColorOrderIndex = 1;
    if I(j+1)-I(j) == 0
        plot(I(j):I(j+1),u(I(j):I(j+1)),'.')    
    else
        plot(I(j):I(j+1),u(I(j):I(j+1)))
    end
end
hold off
end