function [Yhat, YProb] = a_nn_predict(model, test_x)
[Yhat, YProb] = nnpredict_my(model, test_x);
end
