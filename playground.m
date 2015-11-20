%%playground
Y = genders_train;
X = words_train;
% boolean -> 
% X = words_train > 0;
  
%%
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @predict_MNNB);
mean(accuracy)

%%
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'Distance','minkowski', 'Exponent',3, 'NumNeighbors',30),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);

%%
mdl2 = @(train_x,train_y,test_x,test_y) k_means(train_x,train_y,test_x,test_y, 10);
cross_validation(X, Y, 4, mdl2);