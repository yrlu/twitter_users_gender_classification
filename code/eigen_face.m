% Author: Max Lu
% Date: Nov 28

% Inputs: 
%   train_x, train_y, test_x, test_y: training and testing data
%   accuracy: the expected accuracy
%   opts: please pass all your options of the classifier here!

% Outputs: 
%   Yhat: The labels predicted, 1 for female, 0 for male, -1 for uncertain,
%       which means the probability of correctly classification is below 
%       "accuracy" for that sample!
%   YProb: This is all the *RAW* outputs of the classifier.

function [Yhat, YProb] = eigen_face(img_scores_train, train_y, img_scores_test, test_y, accuracy, opts)
pc = 1:100;
% certain_train = certain(1:5000);
train_x = img_scores_train(:, pc);
test_x = img_scores_test(:,pc);
% X = X(logical(certain_train),:);
% Y = train_y(logical(certain_train),:);

B = TreeBagger(300,train_x,train_y, 'Method', 'classification');
[Yhat YProb] = B.predict(test_x);
Yhat = str2double(Yhat);


end