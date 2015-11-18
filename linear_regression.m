function [Yhat] = linear_regression(X,Y,testX)

Wmle = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
Yhat = sigmf(testX*Wmle, [2 0])>0.5;



