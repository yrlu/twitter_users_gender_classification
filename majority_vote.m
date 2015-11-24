function [yhat] = majority_vote(trainX, trainY, testX, testY)
yhat1 = info_gain_tree_ensemble(trainX, trainY, testX, 400);
yhat2 = info_gain_tree_ensemble(trainX, trainY, testX, 1000);
yhat3 = info_gain_tree_ensemble(trainX, trainY, testX, 500);
LogRmodel = train(trainY, sparse(trainX), ['-s 0', 'col']);
% LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), LogRmodel, ['-q', 'col']) - 0.5);
[yhat4, ~,~] = predict(testY, sparse(testX), LogRmodel, ['-q', 'col']);

yhat_mean = (2*yhat1+yhat2+yhat3+yhat4)/5;
yhat = yhat_mean > 0.5;

% 1500/1000/500/ 88.76% 
% 1200/1000/800 88.48%
