function [mixture, optmixture] = GaussianMixtureWithDecorrelation(pixels, initK, finalK, verbose, estKind, ConditionNumber)
% [mixture, optmixture] = GaussianMixtureWithDecorrelation(pixels, initK, finalK, verbose, ConditionNumber)
% performs the EM algorithm to estimate the order, and parameters of a
% Gaussian Mixture model for a given set of observation. The parameters
% are estimated from decorrelated coordinates to condition the
% problem better.
%
%     pixels: a N x M matrix of observation vectors with each row being an
%        M-dimensional observation vector, totally N observations
%     initK: the initial number of clusters to start with and will be reduced
%        to find the optimal order or the desired order based on MDL
%     finalK: the desired final number of clusters for the model.
%        Estimate the optimal order if finalK == 0.
%     estKind = diag constrains the class covariance matrices to be diagonal
%     estKind = full allows the the class covariance matrices to be full matrices
%     ConditionNumber: a constant scaling that controls the ratio of
%        mean to minimum diagonal element of the estimated covariance
%        matrices. The default value is 1e5.
%     verbose: true/false, return clustering information if true
%
%     mixture: an array of mixture structures with each containing the
%        converged Gaussian mixture at a given order
%           mixture(l).K: order of the mixture
%           mixture(l).M: dimension of observation vectors
%           mixture(l).rissanen: converaged MDL(K) 
%           mixture(l).loglikelihood: ln( Prob{Y=y|K, theta*} )
%           mixture(l).cluster: an array of cluster structures with each
%              containing the converged cluster parameters
%                 mixture(l).cluster(k).pb: pi(k)=Prob(Xn=k|K, theta*)
%                 mixture(l).cluster(k).mu: mu(k), mean vector of the k-th cluster
%                 mixture(l).cluster(k).R: R(k), covariance matrix of the k-th cluser
%     optmixture: one of the element in the mixture array. 
%        If finalK > 0, optmixture = mixture(1) and is the mixture with order finalK. 
%        If finalK == 0, optmixture is the one in mixture with minimum MDL
%

if ~isnumeric(initK) || ~all(size(initK)==[1,1]) || initK<=0 || mod(initK,1)~=0
   error('GaussianMixture: initial number of clusters initK must be a positive integer');
end
if ~isnumeric(finalK) || ~all(size(finalK)==[1,1]) || finalK<0 || mod(finalK,1)~=0
   error('GaussianMixture: final number of clusters finalK must be a positive integer or zero');
end
if finalK > initK
   error('GaussianMixture: finalK cannot be greater than initK');
end
if ~isa(pixels,'double')
   pixels = double(pixels);
end
if ~exist('ConditionNumber','var')
    ConditionNumber = 1e5;
end

% decorrelate and normalize the data
smean = mean(pixels,1);
scov = cov(pixels);
[E D] = eig(scov);
T = E*inv(sqrt(D));
invT = inv(T);
tmp = zeros(size(pixels,2));
for i = 1:size(pixels,2)
    tmp(i,i) = smean(i);
end
pixels = (pixels - (tmp*ones(size(pixels')))')*T;
clear tmp

% estimate parameters in decorrelated coordinates
[mixture, optmixture] = GaussianMixture(pixels, initK, finalK, verbose, estKind, ConditionNumber);

% transform the parameters back to original coordinates
if finalK>0
    for k=1:mixture(1).K
        mixture(1).cluster(k).mu = (mixture(1).cluster(k).mu'*invT + smean)';
        mixture(1).cluster(k).R = invT'*mixture(1).cluster(k).R*invT;
        mixture(1).cluster(k).invR = T*mixture(1).cluster(k).invR*T';
        mixture(1).cluster(k).const = mixture(1).cluster(k).const - log(det(invT'*invT))/2;
    end
    optmixture = mixture(1);
else
    for j=1:initK
        for k=1:j
            mixture(j).cluster(k).mu = (mixture(j).cluster(k).mu'*invT + smean)';
            mixture(j).cluster(k).R = invT'*mixture(j).cluster(k).R*invT;
            mixture(j).cluster(k).invR = T*mixture(j).cluster(k).invR*T';
            mixture(j).cluster(k).const = mixture(j).cluster(k).const - log(det(invT'*invT))/2;
        end
    end
    for k=1:optmixture.K
        optmixture.cluster(k).mu = (optmixture.cluster(k).mu'*invT + smean)';
        optmixture.cluster(k).R = invT'*optmixture.cluster(k).R*invT;
        optmixture.cluster(k).invR = T*optmixture.cluster(k).invR*T';
        optmixture.cluster(k).const = optmixture.cluster(k).const - log(det(invT'*invT))/2;
    end
end


   
