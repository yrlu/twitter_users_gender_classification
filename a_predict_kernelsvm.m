%Predict kernel svm
%Input: model, Xtrain(the X that used for training this model), Xtest, kernel
%Ouput: Yhat, Yscore
% Deng, Xiang 12/5/2015
function [ Yhat, Yscore ] = a_predict_kernelsvm( model, Xtrain, Xtest)
kernel =  @(x,x2) kernel_intersection(x, x2);
Ktest = kernel(Xtrain, Xtest);
[Yhat, ~ ,Yscore] = svmpredict(ones(size(Ktest,1),1), [(1:size(Ktest,1))' Ktest], model);
if sum(bsxfun(@times, Yhat, Yscore)) < 0
    Yscore = -Yscore;
end
end

