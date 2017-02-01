% Author: Max Lu
% Date: Nov 20


function [accuracy, Ypredicted, Ytest, at] = cross_validation_adaboost(X, Y, folds, classifier)

[n, ~] = size(X);
[parts] = make_xval_partition(n, folds);
Ypredicted = [];
accuracy = [];
Ytest = [];
at = [];
for i = 1:folds
   trainX = X(parts~=i, :);
   trainY = Y(parts~=i);
   testX = X(parts==i, :);
   testY = Y(parts==i);
   Ytest = [Ytest;testY];
   [Yhat,a] = classifier(trainX, trainY, testX, testY);
   at = [at, a];
   Ypredicted = [Ypredicted;Yhat];
   acc =sum(Yhat==testY)/size(testY,1);
   accuracy = [accuracy, acc];
   i
   acc
end