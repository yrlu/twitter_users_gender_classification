function [Yhat, YProb] = a_logistic_predict(model, test_x)
[Yhat, ~, YProb] = predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']);
if sum(bsxfun(@times, Yhat, YProb)) < 0
    YProb = -YProb;
end
end