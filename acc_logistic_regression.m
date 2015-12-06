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

function [Yhat, YProb, model] = acc_logistic_regression(train_x, train_y, test_x, test_y, accuracy, opts)
disp('Training logistic regression..');
LogRmodel = train(train_y, sparse(train_x), ['-s 0 -q', 'col']);
% LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), LogRmodel, ['-q', 'col']) - 0.5);
% save('./models/LogRmodel.mat', 'LogRmodel');
[Yhat, ~, YProb] = predict(test_y, sparse(test_x), LogRmodel, ['-q', 'col']);
if Yhat(1) == 1 && YProb(1)<0
    YProb = -YProb;
elseif Yhat(1) ==0 && YProb(1)>0
    YProb = -YProb;
end
model = LogRmodel;

end