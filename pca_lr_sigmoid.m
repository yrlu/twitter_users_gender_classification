% Author: Max Lu
% Date: Nov 17

%% Load the data first, see data_preprocess.m

%% 
tic
X = [words_train, image_features_train; words_test, image_features_test];
Y = genders_train;
folds = 8;
[n m] = size(words_train);

% ----
% disp('Generate PCA');
% [coef, scores, eigens] = pca(X);
% plot(cumsum(eigens)/sum(eigens));
% save('coef.mat', 'coef');
% save('scores.mat', 'scores');
% save('eigens.mat', 'eigens');
% ----
toc
% ----
% disp('Load PCA');
% load('coef.mat', 'coef');
% load('scores.mat', 'scores');
% load('eigens.mat', 'eigens');
% ---- 
toc


% ---- Use following code to search for the best number of PC to include
% -- for training:
% disp('Search for the best number of PC');
% acc = []
% for i = 1:80
%     X = scores(1:n, 1:10*i);
%     toc
%     [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, folds, @linear_regression);
%     toc
%     i
%     mean(accuracy)
%     acc = [acc ; mean(accuracy)];
% end
% 
% plot(acc);
% toc


% I found that 320 principal components work best.
disp('linear regression + cross-validation');
X = scores(1:n, 1:320);
toc
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, folds, @linear_regression);
toc
accuracy
mean(accuracy)
toc

% % logistic regression
X = scores(1:n, 1:2000);
addpath('./liblinear');
% disp('logistic regression + cross-validation');
% toc
% [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @logistic);
% toc
% accuracy
% mean(accuracy)
% toc

trainX = scores(1:n, 1:2000);
testX = scores(n+1:size(scores,1), 1:2000);
model = train(Y, sparse(trainX), ['-s 0', 'col']);
[Yhat] = predict(ones(size(testX, 1),1), sparse(testX), model, ['-q', 'col']);

% model = train(train_y, sparse(train_x), ['-s 0', 'col']);
% [Yhat] = predict(test_y, sparse(test_x), model, ['-q', 'col']);


% SVM
% X = scores(1:n, 1:300);
% addpath('./libsvm');
% disp('SVM + Cross-validation');
% [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @svm);
% toc
% accuracy
% mean(accuracy)
% toc
