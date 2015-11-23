function [yhat] = majority_vote(trainX, trainY, testX, testY)
yhat1 = info_gain_tree_ensemble(trainX, trainY, testX, 800);
yhat2 = info_gain_tree_ensemble(trainX, trainY, testX, 1000);
yhat3 = info_gain_tree_ensemble(trainX, trainY, testX, 1200);
yhat_mean = (yhat1+yhat2+yhat3)/3;
yhat = yhat_mean > 0.5;

% 1500/1000/500/ 88.76% 
% 1200/1000/800 88.48%
