function [Yhat, Yprob] = svm_predict( train_x, train_y, test_x, test_y )
addpath ./libsvm
%  sprintf('-t 2 -c %g',)
model = svmtrain(train_y, train_x, '-t 2 -c 100');
[Yhat acc Yprob] = svmpredict(test_y, test_x, model);
end