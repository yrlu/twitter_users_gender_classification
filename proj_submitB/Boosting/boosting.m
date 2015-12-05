% Deng, Xiang 11/28/2015
% Bagging doesn't work for Naive Bayes, 1.in practice taking
% subsets i.i.d (resampling with replacement) and train seperate
% naive bayes will results in very similar classifiers,
% 2. combining those NBs by averaging the parameters has similar effect
% to taking the majority vote from the classifiers, this also behaves like training NB
% on the full dataset; this is probably because NB is a counting models and given
% enough data, combining mulitple models and adding counts together
% is close to the true statistical
% models (or the full counting model formed from the entire data set)

% The motivates me to investigate adaboost, instead of taking samples iid,
% reweight xns and see what we can learn...it turns out adding multiple NB
% is just like one big NB, this boosting process also significantly
% increase the NB bias, adding trees may help...

% Note Y here is +-1 instead of 1 and 0
function [ a,models ] = boosting( learner,predictor, X, Y ,T, opt)
D = ones(size(X,1),1)/size(X,1); % 1/n , n by 1 vector
Z = ones(T,1);
a = ones(T,1);
e =ones(T,1); 

for t = 1:T
    indices=randsample(1:size(D,1),size(D,1),true,D); % weighted random sample of x with replacement
    model_cur=learner(X(indices,:),Y, opt);
    models.mdl{t}=model_cur;
    Yhat=predictor(model_cur,X);
    e(t)=(Yhat~=Y)'*D;
    a(t)=1/2*log((1-e(t))/e(t));
    expvec=exp(-a(t)*Y.*Yhat);
    Z(t)=D'*expvec;
    D=D.*expvec/Z(t);
end


end

