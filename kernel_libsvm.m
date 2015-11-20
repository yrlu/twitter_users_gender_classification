function [test_err info] = kernel_libsvm(X, Y, Xtest, Ytest)
% Trains a SVM using libsvm and evaluates on test data.
%
% Usage:
%
%   [TEST_ERR INFO] = KERNEL_LIBSVM(X, Y, XTEST, YTEST)
%
% Runs training and testing of a SVM with the given kernel function, using
% cross validation to choose regularization parameter C. X, Y, XTEST, and
% YTEST should be created using MAKE_SPARSE. 

% Use built-in libsvm cross validation to choose the C regularization
% parameter.
crange = 2.^[-4:2:8];
for i = 1:numel(crange) %% 10 fold CV, RBF
    acc(i) = svmtrain(Y, X, sprintf('-t 2 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

% Train and evaluate SVM classifier using libsvm
model = svmtrain(Y, X, sprintf('-t 2 -c %g', crange(bestc)));
[yhat acc vals] = svmpredict(Ytest, Xtest, model);
test_err = mean(yhat~=Ytest);

% Optionally we can look at more information from training/testing.
%info.vals = vals;
%info.yhat = yhat;
%info.model = model;




    
    
