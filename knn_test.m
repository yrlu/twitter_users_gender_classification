function [testLabels] = knn_test(K, trainPoints, trainLabels, testPoints, distFunc)
%KNN_TEST - Evaluates KNN predictions given training data and parameters.
% 
%  [testLabels] = knn_test(K, trainPoints, trainLabels, testPoints, ...
%                          [distFunc])
%
%   K - Number of nearest neighbors to use
%   trainPoints - N x P matrix of examples, where N = number of points and
%       P = dimensionality
%   trainLabels - N x 1 vector of labels for each training point.
%   testPoints  - M x P matrix of examples, where M = number of test points
%       and P = dimensionality
%   distFunc - OPTIONAL string declaring which distance function to use:
%       valid functions are 'l2','l1', and 'linf'
%
%   Returns an M x 1 vector that is the average of the training labels of
%   the K nearest neighbors to each test point using the given distance
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

% Now we have a M x 1 x N 3-D array of distances between each pair of points.
% To find the nearest neighbors, we first "squeeze" this to become a M x N
% matrix, and then sort within each of the M rows separately. Note that we
% use only the second output from the "sort" command.

distMat = squeeze(distMat);
if numTestPoints == 1 % if only 1 point, squeeze converts to col vector
    distMat = distMat';
end

[sorted nnMat] = sort(distMat, 2);

nnMat = nnMat(:,1:K);

% Average over the nearest neighbors to get predicted labels
testLabels = trainLabels(nnMat);
if size(nnMat, 1) == 1 % again, if only 1, gets incorrectly converted to col vector
    testLabels = testLabels';
end

if K > 1
    testLabels = mean(testLabels, 2);
end
