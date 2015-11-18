function [Yhat] = linear_regression(X,Y,testX)

% [n m] =size(words_train);

% ntrain = int32(n*3/4);

% X = words_train(1:ntrain, :);
% Y = genders_train(1:ntrain, :);

% testX = words_train(ntrain+1:n, :);
% testY = genders_train(ntrain+1:n, :);

[coef, scores, eigens] = pca([X;testX]);
[n m] = size(X);
X = scores(1:n, 1:500);
Wmle = inv(X'*X+eye(size(X,2))*1e-3) * (X')* Y;
testX = scores(n+1:size(scores,1), 1:500);
Yhat = sigmf(testX*Wmle, [2 0])>0.5;

% accuracy = sum(Yhat == testY)/size(testY, 1)



