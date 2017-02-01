%Deng, Xiang
% 11/20-
function step = update_constant(X_i, y_i, w, Rate)
% Computes the update step given a constant learning rate Rate
%
% Usage:
%     step = update_constant(X_i, y_i, w, Rate)
%
% Takes a 1 x (D+1) matrix X_i representing the current example,
% a scalar +1/-1 y_i correct label for that example
% the current (D+1)x1 weights of the perceptron,
% and a constant learning rate parameter (Eta in the notes)
%
% Returns the magnitude of the step in the direction of X_i that should be
% taken by perceptron

%% YOUR CODE GOES HERE
% This should be a very simple one liner
step=Rate*(y_i-sign(X_i*w));
