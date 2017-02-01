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
%   YProb: This is all the *RAW* outputs of the classifier

function [Yhat, YProb] = ecoc(train_x, train_y, test_x, test_y, accuracy, opts)
    mdl = fitcecoc(train_x,train_y);
    [Yhat,YProb] = predict(mdl,test_x);
end