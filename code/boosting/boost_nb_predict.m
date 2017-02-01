% Deng, Xiang 2015/11/28
function [ Yhat ] = boost_nb_predict( model,Xtest)
    P_priors = predict_fastnb(model, sparse(Xtest)');
    [~, Yhat] = max(P_priors, [], 2);
    Yhat(Yhat==1)=-1;
    Yhat(Yhat==2)= 1;
end

