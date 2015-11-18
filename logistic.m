% Author: Max Lu
% Date: Nov 17

function [Yhat] = logistic(train_x, train_y, test_x)
    model = train(train_y, sparse(train_x), ['-s 0', 'col']);
    [Yhat] = predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']);
end

