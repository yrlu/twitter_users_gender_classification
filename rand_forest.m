% Author: Max Lu
% Date: Nov 21


% Assuming we have all the data loaded into memory.
function [Yhat] = rand_forest(train_x, train_y, test_x, test_y)
    B = TreeBagger(150,train_x,train_y, 'Method', 'classification');
%     RFpredict = @(test_x) sign(str2double(B.predict(test_x)) - 0.5);
    Yhat = str2double(B.predict(test_x));
end