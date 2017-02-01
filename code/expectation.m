function [llh,post] = expectation(X, model)
% llh = number: loglikelihood of data: data belongs to this distribution.
% post = n-by-k matrix; post(i,j) = posterior probability of data i belongs
% to cluster j

mu = model.mu; % mean
Sigma = model.Sigma; % covariance
w = model.PComponents; % weights for each cluster; pb(k) = Prob(Xn = k|K, theta*)

n = size(X,1); % number of rows
k = size(mu,1); % number of components 

post = zeros(n,k); % prob i in j
logRho = zeros(n,k); % log prob

% assume shared and diagonal covariance.
% compute log of component conditional density weighted by the component
%   probability: log Prob(Xn=k|Yn=yn, theta)

L = sqrt(Sigma); % L * L' = Sigma
det = sum(log(Sigma));
for i = 1:k
    logRho(:,i) = loggausspdf(X, mu(i,:), w(i), L, det);
end

llmax = max(logRho, [], 2); % max of each row (llcluster) 
% minus max to avoid underflow
post = exp(bsxfun(@minus, logRho, llmax));
dens = sum(post,2); %density function
% normalize post
post = bsxfun(@rdivide, post, dens);
logpdf = log(dens) + llmax;
llh = sum(logpdf);

end %estep
    
function y = loggausspdf(X, mu, w, L, det)
% log prob of i in j
%
% inspired by matlab wdensity & emgm by Chen Mo.

d = size(X,2); %dimensions
X = bsxfun(@minus,X,mu); % centered

xRinv = bsxfun(@times, X, (1./L)); % prep mahalanobis distance: i from j
distance = sum(xRinv.^2, 2); %distance

c = d*log(2*pi) + det - 2* log(w);   % normalization constant
y = -(c+distance)/2;

end %loggausspdf
    
    
    