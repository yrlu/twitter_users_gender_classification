% Author: Dongni Wang
% Date: Nov 20
% Multinomial Naive Bayes


function [yhat] = applyMNNB(prior, condprob, indexNotN, Xtest)

lTest = size(Xtest,1);

preSumA = log(Xtest(:,indexNotN).*repmat(condprob,lTest,1));
preSumB = log(Xtest(:,indexNotN).*repmat(1-condprob,lTest,1));
preSumA(isinf(preSumA)) = 0;
preSumB(isinf(preSumB)) = 0;
scoreA =  sum(preSumA,2) + log(prior);
scoreB =  sum(preSumB,2) +log(1- prior);
AminusB = scoreA-scoreB;
yhat = AminusB > 0;
end

