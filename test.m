% Author: Max Lu
% Date: Nov 17

Xtrain  = words_train;
[n,m] = size(Xtrain);
% X = scores; % +++ PCA 
% Y = genders_train;
% Conduct simple scaling on the data [-1,1]
Xnorm = (Xtrain - repmat(min(Xtrain),n,1))./repmat(range(Xtrain),n,1);
% Caution, remove uninformative NaN data % for nan - columns
Xcl = Xnorm(:,all(~isnan(Xnorm)));
%% 
X = scores(1:n, 1:320);
%X = Xcl;
folds = 5;
Y=genders_train;
accuracy = [];

%%
% [n ~] = size(X);
[parts] = make_xval_partition(n, folds);

for i = 1:folds
   trainX = X(parts~=i, :);
   trainY = Y(parts~=i);
   testX = X(parts==i, :);
   testY = Y(parts==i);
%    Ytest = [Ytest;testY];
%    [Yhat] = classifier(trainX, trainY, testX);
%    Ypredicted = [Ypredicted;Yhat];

   NBModel = fitNaiveBayes(trainX,trainY); %,'Distribution','mvmn')
   Yhat = predict(NBModel,testX);
   
%     [ConfusionMat1,labels] = confusionmat(Y,predictLabels1)
   accuracy = [accuracy, sum(Yhat==testY)/size(testY,1)];
end

accuracy
mean(accuracy)




%% Linear Regression + Sigmoid Function + PCA  70%
 [accuracy, Ypredicted, Ytest] = cross_validation(words_train, genders_train, 5, @logistic);

 