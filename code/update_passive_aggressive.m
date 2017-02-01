%Deng, Xiang
% 11/20-
function step = update_passive_aggressive(X_i, y_i, w)
% Computes the update step using a passive aggressive approach
%
% Usage:
%     step = update_passive_aggressive(X_i, y_i, w)
%
% Takes a 1 x (D+1) matrix X_i representing the current example,
% a scalar +1/-1 y_i correct label for that example
% and the current (D+1)x1 weights of the perceptron as arguments
%
% Returns the magnitude of the step in the direction of X_i that should be
% taken by the perceptron


%% YOUR CODE GOES HERE
L=0;
if (y_i *X_i *w <1)
    L=1-y_i *(X_i *w);
end
step=L/(norm(X_i,2)^2)*y_i;