%% Multinomial Naive Bayes model 
% Ref: http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf

function [prior,condprob] = trainMNNB(Xtrain, Ytrain)
% train a multinomial Naive Bayes model
% return prior of each class and conditional probability of word|class
N = size(Xtrain,1);
% for each class, compute prior 
prior = sum(Ytrain)/N; %for Ytrain = 1

% Binarized (Boolean) Multinomial Naive Bayes model; Also the Bernoulli
% documment model
Xtrain(Xtrain>0) = 1;

%Multinomial Naive Bayes model 
countTermA = sum(Xtrain(logical(Ytrain),:));
countTermB = sum(Xtrain(~logical(Ytrain),:));
countAllA = sum(countTermA);
countAllB = sum(countTermB);

% for each term compute conditional probability
condprobA = (countTermA+1)./(countAllA+N);
condprobB = (countTermB+1)./(countAllB+N);
condprob = [condprobA; condprobB];