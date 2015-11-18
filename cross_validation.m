% Author: Max Lu
% Date: Nov 17


function [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, folds, classifier)

[n ~] = size(X);
[parts] = make_xval_partition(n, folds);
Ypredicted = [];
accuracy = [];
Ytest = [];
for i = 1:folds
   trainX = X(parts~=i, :);
   trainY = Y(parts~=i);
   testX = X(parts==i, :);
   testY = Y(parts==i);
   Ytest = [Ytest;testY];
   [Yhat] = classifier(trainX, trainY, testX, testY);
   Ypredicted = [Ypredicted;Yhat];
   accuracy = [accuracy, sum(Yhat==testY)/size(testY,1)];
end