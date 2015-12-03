% Deng, Xiang 11/28/2015
function [ Yhat ] = predict_nb_fast(model, Xtest )
P_priors = (model, sparse(Xtest)');
[~, Yhat] = max(P_priors, [], 2);
Yhat=Yhat-1;

end

