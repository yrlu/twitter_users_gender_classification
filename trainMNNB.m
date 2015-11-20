%% play with Multinomial Naive Bayes model 
function [prior,condprob, indexNotN] = trainMNNB(Xtrain, Ytrain)
% train a multinomial Naive Bayes model
% return prior of each class and conditional probability of word|class
N = size(Xtrain,1);
%classA = find(Ytrain); 
%classB = find(~Ytrain);
% for each class, compute prior 
prior = sum(Ytrain)/N %for Ytrain = 1
%priorB = 1-priorA; %for Ytrain = 0
%prior = [priorA, priorB]; 
% 

countTermA = sum(Xtrain(logical(Ytrain),:));

%countTermB = sum(Xtrain(classB,:));
countAll = sum(Xtrain);
% for each term compute conditional probability
condprob = countTermA./countAll;
%condprobB = 1 - condprobA;
indexNotN = ~isnan(condprob);
condprob = condprob(:,indexNotN);
%condprobB = condprobB(:,(~isnan(condprobB)));
% conditional probability of word given a class = relative frequency of
% word in doc belonging to class. 

%%
%countTermA = sum(Xtrain(logical(Ytrain),:)>0);
%condprob = (countTermA+1)./(N+2);