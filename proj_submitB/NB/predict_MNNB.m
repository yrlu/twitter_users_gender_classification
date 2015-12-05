% Author: D.W 

function [yhat, ys] = predict_MNNB(trainX, trainY, testX, testY)
% very Naive Bayes Text Classifier (2 classes) 
[prior, condprob] = trainMNNB(trainX, trainY);
% Binarized (Boolean) Multinomial Naive Bayes model
testX(testX>0) = 1;
[yhat, ys] = applyMNNB(prior, condprob, testX);

end