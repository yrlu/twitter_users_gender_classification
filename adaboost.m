% Author: Max Lu
% Date: Nov 18

function [Yhat] = adaboost(train_x, train_y, test_x, test_y)
    ClassTreeEns = fitensemble(train_x,train_y,'LogitBoost',200,'Tree');
    Yhat = predict(ClassTreeEns,test_x);
end