function [testLabels] = kernreg_test(sigma, trainPoints, trainLabels, testPoints, distFunc)
%KERNREG_TEST - Evaluates kernel regression/classification predictions given training data and parameters.
% 
%  [testLabels] = kernreg_test(sigma, trainPoints, trainLabels, testPoints, ...
%                          [distFunc])
%
%   sigma - Kernel width parameter
%   trainPoints - N x P matrix of examples, where N = number of points and
%       P = dimensionality
%   trainLabels - N x 1 vector of labels for each training point.
%   testPoints  - M x P matrix of examples, where M = number of test points
%       and P = dimensionality
%   distFunc - OPTIONAL string declaring which distance function to use:
%       valid functions are 'l2','l1', and 'linf'
%
%   Returns an M x 1 vector that is the weighted average of the training labels of
%   according to a guassian kernel with width sigma and the given distance
%   function. Note that it is up to you to interpret these averages as
%   either the sign of the classification (for binary classifiation) or the
%   average prediction (for regression).

if nargin<5
    distFunc = 'l2';
end

% NOTE: this code is heavily VECTORIZED, which means that it does not use a
% any "for" loops and runs very quickly. Understanding this code is a
% good exercise for learning how to write programs in Matlab that run very
% fast.

numTestPoints = size(testPoints, 1);
numTrainPoints = size(trainPoints, 1);

% The following lines compute the difference between every test point and
% every train point in each dimension separately, using a single M x P X N
% 3-D array subtraction:

% Step 1:  Reshape the N x P training matrix into a 1 X P x N 3-D array
trainMat = reshape(trainPoints', [1 size(trainPoints,2) numTrainPoints]);
% Step 2:  Replicate the training array for each test point (1st dim)
trainCompareMat = repmat(trainMat, [numTestPoints 1 1]);
% Step 3:  Replicate the test array for each training point (3rd dim)
testCompareMat = repmat(testPoints, [1 1 numTrainPoints]);
% Step 4:  Element-wise subtraction
diffMat = testCompareMat - trainCompareMat;

% Now we can compute the distance functions on these element-wise
% differences:
if strcmp(distFunc, 'l2')
    distMat = sqrt(sum(diffMat.^2, 2));
elseif strcmp(distFunc, 'l1')
    distMat = sum(abs(diffMat), 2);
elseif strcmp(distFunc, 'linf')
    distMat = max(abs(diffMat), [], 2);
else
    error('Unrecognized distance function');
end

% Now we have a M x 1 x N 3-D array of distances between each pair of
% points. We squeeze this to a M x N matrix, then use these distances to 
% compute the corresponding M x N kernel matrix:

distMat = squeeze(distMat);
if numTestPoints == 1 % squeeze will make this a column vector if only 1 point
    distMat = distMat';
end

kernMat = exp(-distMat.^2/sigma.^2);

% Next, replicate the training label matrix to become M x N:
trainLabels = repmat(trainLabels', numTestPoints, 1);
% Finally, compute a weighted average over the M rows using the kernel:
testLabels = sum(trainLabels.*kernMat,2)./sum(kernMat,2);
