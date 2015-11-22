function obj = fitGMM (X, k, cvType)
% Fit a Gaussian mixture distribution to data.
% k = number of components
% X = the data to be fitted (samples-by-features)
% cvType = 0: 'full', 1: 'diagonal'
% S.mu, S.Sigma, S.pb
% 
% MaxIter = 200; SharedCovariance = true, Start by kmeans
% Using Expectation-Maximization (EM) algorithm.
% 
%
[n, d] = size(X); % X is n-by-d matrix
obj = model;
obj.NComponents = k;
obj.NDimensions = d;
obj.NData = n;
obj.Converged = false; % initialize obj
obj.Iters = 0;
obj.NlogL = 0;


if cvType == 1
	obj.CovType = 'diagonal';
else % type = 0
	obj.CovType = 'full';
end

% Rows of X with NaNs are excluded from the fit. Remove NaNs from X
wasnan = any(isnan(X),2);
hadNaNs = any(wasnan);
if hadNaNs
  %  warning(message('stats:gmdistribution:MissingData'));
    X = X(~wasnan,:);
end

% initialize using k-means++
ini = kmeansplus(X, k, cvType);% initial parameter
%  S.mu, S.Sigma, S.pb

% try learning cluster 


S = ini; %initial parameter
ll_old = -inf; % last llh

tol = 1e-6; %tolerance
%llh = -inf(1,200); %loglikelihood 

for iter = 1:200
    % use EM algorithm 
	% estep : calculate membership weights 
	% mstep : use membership weights to calculate new parameter values
    
    % estep
    [ll, post] = expectation(X,S); % calculate weights & loglikehood of data
    
    %check convergence
    llDif = ll - ll_old;
    if(llDif >= 0 && llDif < tol*abs(ll))
        obj.Converged = true; % ll: max log-likelihood: ln(Prob{Y=y|K, theta*} 
        break;
    end
    ll_old = ll; 
    
    %mstep: idea from matlab gmcluster
    S.PComponents = sum(post,1); % old weight
    
    S.Sigma = zeros(1,d);
    for i = 1:k % for each cluster
        S.mu(i,:) = post(:,i)' * X / S.PComponents(i); 
        Xcentered = bsxfun(@minus, X, S.mu(i,:)); % distance from mean
        S.Sigma = S.Sigma + post(:,i)' *(Xcentered.^2);  % variance
    end
    S.Sigma = S.Sigma/sum(S.PComponents);
   
    % normalize PComponents
    S.PComponents = S.PComponents/sum(S.PComponents);
    
end % end iteration
obj.Iters = iter;
obj.mu = S.mu;
obj.Sigma = S.Sigma;
obj.PComponents = S.PComponents;
obj.NlogL = -ll;

if ~obj.Converged
    fprintf('Not converged in %d steps.\n',200);
end

end

	











