
function [Xcl] = norml(X)
sizeX= size(X,1);
Xnorm = (X - repmat(min(X),sizeX,1))./repmat(range(X),sizeX,1);
% Caution, remove uninformative NaN data % for nan - columns
Xnorm(isnan(Xnorm)) = rand(size(Xnorm(isnan(Xnorm)), 1), 1);
% Xcl = Xnorm(:,all(~isnan(Xnorm)));  
Xcl =Xnorm;
end

