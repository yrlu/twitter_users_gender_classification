% Author: Max Lu
% Date: Nov 18


% Assuming we have all the data loaded into memory.
function [Yhat] = adaboost(train_x, train_y, test_x, test_y)
%     ClassTreeEns = fitensemble(train_x,train_y,'LogitBoost',200,'Tree');
%     Yhat = predict(ClassTreeEns,test_x);
    
% first train a linear regression classifier
X = train_x;
Y = train_y;
Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
LRpredict = @(test_x) sigmf(test_x*Wmap, [2 0])>0.5;
% Yhat = sigmf(test_x*Wmap, [2 0])>0.5;
% Just call: linear_regression(train_x,train_y,test_x,test_y);

% train a logistic regression classifier
% trainX = scores(1:n, 1:3200);
% testX = scores(n+1:size(scores,1), 1:3200);

model = train(Y, sparse(X), ['-s 0', 'col']);
% [Yhat] = predict(ones(size(testX, 1),1), sparse(testX), model, ['-q', 'col']);

Yhat = 0.5*LRpredict(test_x)+0.5*predict(test_y, sparse(test_x), model, ['-q', 'col']);
Yhat = uint8(Yhat);
end