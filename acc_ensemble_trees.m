% Author: Max Lu
% Date: Nov 23

% Inputs: 
%   train_x, train_y, test_x, test_y: training and testing data
%   accuracy: the expected accuracy
%   opts: please pass all your options of the classifier here!

% Outputs: 
%   Yhat: The labels predicted, 1 for female, 0 for male, -1 for uncertain,
%       which means the probability of correctly classification is below 
%       "accuracy" for that sample!
%   YProb: This is all the *RAW* outputs of the classifier.

function [Yhat, YProb] = acc_ensemble_trees(train_x, train_y, test_x, test_y, accuracy, opts)
ens = fitensemble(train_x,train_y,'LogitBoost',200,'Tree' ); 
% FSPredict = @(test_x) sign(predict(ens,test_x)-0.5);
[Yhat, YProb]= predict(ens,test_x);
end