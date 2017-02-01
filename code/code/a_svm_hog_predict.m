function [Yhat, Yprob] = a_svm_hog_predict(model, test_x)
addpath('./libsvm')
[Yhat acc Yprob] = svmpredict(ones(size(test_x,1),1), test_x, model);
if sum(bsxfun(@times, Yhat, Yprob)) < 0
    Yprob = -Yprob;
end
end