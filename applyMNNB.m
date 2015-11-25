% Author: Dongni Wang
% Date: Nov 20
% Multinomial Naive Bayes


function [yhat, y_scores] = applyMNNB(prior, condprob, Xtest)

% Multinomial Naive Bayes model
lTest = size(Xtest,1);

preSumA = log(Xtest.*repmat(condprob(1,:),lTest,1));
preSumB = log(Xtest.*repmat(condprob(2,:),lTest,1));
preSumA(isinf(preSumA)) = 0;
preSumB(isinf(preSumB)) = 0;

% This part is for Bernoulli model 
% Also, we want to take the unused words into account.
% first make the Xtest 0 -> 1; 1 -> 0 for easier calculation
Xtest = Xtest == 0; % or use 1- Xtest
unpreSumA = log(Xtest.*repmat(1-condprob(1,:),lTest,1));
unpreSumB = log(Xtest.*repmat(1-condprob(2,:),lTest,1));
unpreSumA(isinf(unpreSumA)) = 0;
unpreSumB(isinf(unpreSumB)) = 0;
% End Bernoulli session 

scoreA =  sum(preSumA,2) + log(prior);
scoreB =  sum(preSumB,2) +log(1- prior);

% Bernoulli...
scoreA = scoreA + sum(unpreSumA,2);
scoreB = scoreB + sum(unpreSumB,2);

AminusB = scoreA-scoreB;
yhat = AminusB > 0;
%y_scores = [scoreA,scoreB ]
total = scoreA + scoreB;
y_scores = [scoreB./total,scoreA./total];
end

