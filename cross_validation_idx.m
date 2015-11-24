% Author: Max Lu
% Date: Nov 22

% Assuming that we have access to the data in the function, we pass idx to
% the classifiers.
function [accuracy, Ypredicted, Ytest] = cross_validation_idx(n, folds, classifier)
[parts] = make_xval_partition(n, folds);
Ypredicted = [];
accuracy = [];
Ytest = [];
for i = 1:folds
   [Yhat, testY] = classifier(parts==i);
   Ytest = [Ytest;testY];
   Ypredicted = [Ypredicted;Yhat];
   acc =sum(Yhat==testY)/size(testY,1);
   accuracy = [accuracy, acc];
   disp('current fold:');
   i
   disp('accuracy:');
   acc
end



end