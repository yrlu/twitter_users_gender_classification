function nlogl = classNlogl(dist, X)
% returns nlogn, the negative likelihood of the data X given the model dist.
%
% Rows of X with NaNs are excluded from the fit. Remove NaNs from X 
% from matlab function
wasnan = any(isnan(X),2);
hadNaNs = any(wasnan);
if hadNaNs
   % warning(message('stats:gmdistribution:MissingData'));
    X = X(~wasnan,:);
end

%estep
%m.mu = dist.mu % mean
%m.Sigma = dist.Sigma; % covariance
%m.pb = dist.PComponents; % weights for each cluster; pb(k) = Prob(Xn = k|K, theta*)

nlogl = -expectation(X, dist);