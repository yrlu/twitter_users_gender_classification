function [Yhat] = linear_regression(X,Y,testX)

% [n m] =size(words_train);

% ntrain = int32(n*3/4);

% X = words_train(1:ntrain, :);
% Y = genders_train(1:ntrain, :);

% testX = words_train(ntrain+1:n, :);
% testY = genders_train(ntrain+1:n, :);

Wmle = inv(X'*X+eye(size(X,2))*1e-3) * (X')* Y;
Yhat = testX*Wmle>0.5;

% accuracy = sum(Yhat == testY)/size(testY, 1)



