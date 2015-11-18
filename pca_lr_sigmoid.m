% Author: Max Lu
% Date: Nov 17


%% Load the data first, see data_preprocess.m
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');
load('coef.mat', 'coef');
load('scores.mat', 'scores');
load('eigens.mat', 'eigens');

% add lib path:
addpath('./liblinear');

%% 
tic
X = [words_train, image_features_train; words_test, image_features_test];
% X = normc(X);
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

% plot(acc);
% toc


% I found that 320 principal components work best.
disp('linear regression + cross-validation');
X = scores(1:n, 1:320);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, folds, @linear_regression);
accuracy
mean(accuracy)
toc

% Linear regression model
% Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
% Yhat = sigmf(testX*Wmap, [2 0])>0.5;

% % logistic regression

% X = scores(1:n, 1:3200);
% addpath('./liblinear');
% disp('logistic regression + cross-validation');
% [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 8, @logistic);
% accuracy
% mean(accuracy)
% toc



% Search for the best PC number :

% disp('Search for the best PC number : logistic regression');
% acc = []
% for i = 1:6
% X = scores(1:n, 1:2400+i*200);
% [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @logistic);
% accuracy;
% 2400+i*200
% mean(accuracy)
% acc = [acc;mean(accuracy)];
% toc
% end
% plot(acc);

% LogisticModel
% model = train(train_y, sparse(train_x), ['-s 0', 'col']);
% [Yhat] = predict(test_y, sparse(test_x), model, ['-q', 'col']);




% adaboost:
% disp('adaboost + logistic regression ');
% X = scores(1:n, 1:2000);
% ClassTreeEns = fitensemble(X,Y,'LogitBoost',200,'Tree');
% rsLoss = resubLoss(ClassTreeEns,'Mode','Cumulative');
% plot(rsLoss);
% xlabel('Number of Learning Cycles');
% ylabel('Resubstitution Loss');

% 
% X = scores(1:n, 1:500);
% disp('adaboost + logistic regression + cross-validation');
% [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @adaboost);
% accuracy
% mean(accuracy)
% toc



trainX = scores(1:n, 1:3200);
testX = scores(n+1:size(scores,1), 1:3200);
model = train(Y, sparse(trainX), ['-s 0', 'col']);
[Yhat] = predict(ones(size(testX, 1),1), sparse(testX), model, ['-q', 'col']);
% dlmwrite('submit.txt',Yhat,'\n');


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
