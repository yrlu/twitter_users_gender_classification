function predictions = make_final_prediction(model,X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model
predictions = X_test(:,1:5000) * model.w_ridge;
predictions(predictions > 0.5) = 1;
predictions(predictions <= 0.5) = 0;

