% Author: D.W 

function [yhat2, ys2] = predict_MNNB(trainX, trainY, testX, testY)
% very Naive Bayes Text Classifier (2 classes) 
[prior, condprob] = trainMNNB(trainX, trainY);
% Binarized (Boolean) Multinomial Naive Bayes model
testX(testX>0) = 1;

[yhat, ys] = applyMNNB(prior, condprob, testX);

disp('reconstruction error..');
[yr, ~] = applyMNNB(prior, condprob, trainX);
sum(yr == trainY)/size(trainY,1)
%disp('apply EM...');
absDif = abs(ys(:,1)-ys(:,2));
idx = (absDif > 0);
plusX = testX(idx,:);
plusY = testY(idx);
%disp('accu of data we applied to train: ');
%sum(yhat(idx) == plusY)/size(plusY,1)
[p2, cond2] = trainMNNB([trainX;plusX], [trainY;plusY]);
[yhat2, ys2] = applyMNNB(p2, cond2, testX);
%disp('accu before EM');
%sum(yhat == testY)/size(testY,1)
%disp('accu after EM: ');
%sum(yhat2 == testY)/size(testY,1)
%disp('finish apply EM.');
end