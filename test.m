




folds = 5;
X=words_train;
Y=genders_train;
accuracy = [];
[n ~] = size(X);
[parts] = make_xval_partition(n, folds);

for i = 1:folds
   trainX = X(parts~=i, :);
   trainY = Y(parts~=i);
   testX = X(parts==i, :);
   testY = Y(parts==i);
%    Ytest = [Ytest;testY];
%    [Yhat] = classifier(trainX, trainY, testX);
%    Ypredicted = [Ypredicted;Yhat];

    NBModel = fitNaiveBayes(trainX,trainY,'Distribution','mvmn')
    Yhat = predict(NBModel,testX);
    Yhat = Yhat <= 0.5;
%     [ConfusionMat1,labels] = confusionmat(Y,predictLabels1)
   accuracy = [accuracy, sum(Yhat==testY)/size(testY,1)];
end

accuracy
mean(accuracy)

