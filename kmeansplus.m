function param = kmeansplus(X, k, cvType )
% initilize parameter using kmeans++:
% randomly select the first centroid
% repeatedly select subsequent centers with a probability 
% proportional to the distance from itself to the closest center
% that has been chosen.
% cvType = 0: 'full', 1: 'diagonal'
%
% explanation from matlab: gmcluster
%

param.PComponents = ones(1,k)/k; % starting with equal mixing proportions: sum up to 1

iniVar = var(X);
inisigma = diag(iniVar); %covariance 

if cvType == 1 
	Sigma = iniVar; %shared, diagonal
else 
	Sigma = inisigma; 
end
param.Sigma = Sigma;

% select first seed randomly
seedIndex = zeros(k,1);
[C(1,:), seedIndex(1)] = datasample(X,1); %C: centers

% select the other centers by a probabilistic model
for i = 2:k
	sProb = min(distfun(X, C(1:i-1,:), inisigma), [], 2);
	denom = sum(sProb);
	if denom == 0 || denom == Inf
		C(i:k,:) = datasample(X, k-i+1, 1, 'Replace', false);
		break;
	end
	sProb = sProb/denom; % normalize
	[C(i,:), seedIndex(i)] = datasample(X, 1, 1, 'Replace', false, 'Weights', sProb);
end

param.mu = C;
end  % kmeans plus

function dist = distfun(X, C, sigma)
n = size(X,1);
comp = size(C,1);
dist = zeros(n,comp);
for ii = 1:comp
	delta = X - C(repmat(ii,n,1),:); % distance from center
	dist(:,ii) = sum(delta/sigma.*delta,2); % calculate matrix distance to each center
end

end % helper function distance calculator


