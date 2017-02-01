function yhat = random_forest(trainX, trainY, testX, testY) 
B = TreeBagger(90,trainX,trainY, 'Method', 'classification');
yhat = str2double(B.predict(testX));