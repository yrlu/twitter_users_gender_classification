% Author: Max Lu
% Date: Nov 23

% This function is compatible with cross_validation.m

% Inputs: 
%   train_x, train_y, test_x, test_y: training and testing data
%   accuracy: the expected accuracy
%   opts: please pass all your options of the classifier here!

% Outputs: 
%   Yhat: The labels predicted, 1 for female, 0 for male, -1 for uncertain,
%       which means the probability of correctly classification is below 
%       "accuracy" for that sample!
%   YProb: This is all the *RAW* outputs of the neural network.

function [Yhat, YProb] = nn_load_predict(train_x, train_y, test_x, test_y, accuracy, opts)

load('nn.mat');
% [Yhat_t prob_t] = nnpredict_my(nn, train_x);
% sum(~(Yhat_t-1) ~= Y)/size(train_y,1);
[Yhat YProb] = nnpredict_my(nn, test_x);
Yhat = ~(Yhat-1);

end
