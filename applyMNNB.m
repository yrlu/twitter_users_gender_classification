% Author: Dongni Wang
% Date: Nov 20
% Multinomial Naive Bayes


function [yhat, y_scores] = applyMNNB(prior, condprob, Xtest)

lTest = size(Xtest,1);

preSumA = log(Xtest.*repmat(condprob(1,:),lTest,1));
preSumB = log(Xtest.*repmat(condprob(2,:),lTest,1));
preSumA(isinf(preSumA)) = 0;
preSumB(isinf(preSumB)) = 0;
scoreA =  sum(preSumA,2) + log(prior);
scoreB =  sum(preSumB,2) +log(1- prior);
AminusB = scoreA-scoreB;
yhat = AminusB > 0;
%y_scores = [scoreA,scoreB ]
total = scoreA + scoreB;
y_scores = [scoreB./total,scoreA./total];
end

