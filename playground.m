%% data loading:
%% Load the data first, see data_preprocess.m

% KNN
mdl = @(trainX, trainY, testX, testY) sign(predict(fitcknn(trainX,trainY, 'NumNeighbors',11),testX)-0.5);

%%
mdl2 = @(trainX, trainY, testX, testY) sign(predict(fitNaiveBayes(trainX,trainY),testX)-0.5); %,'Distribution','mvmn')
  
%%
if exist('genders_train','var')~= 1
prepare_data;
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');
end

% Prepare/Load PCA-ed data,  
if exist('eigens','var')~= 1
    if exist('coef.mat','file') ~= 2 
        X = [words_train, image_features_train; words_test, image_features_test]; 
        [coef, scores, eigens] = pca(X);
        save('coef.mat', 'coef');
        save('scores.mat', 'scores');
        save('eigens.mat', 'eigens');
    else 
        load('coef.mat', 'coef');
        load('scores.mat', 'scores');
        load('eigens.mat', 'eigens');
    end
end


%% playground
Y = genders_train;
X = words_train;
% boolean -> 
% X = words_train > 0;

%% normalization
% X = [words_train, image_features_train; words_test, image_features_test];

% Conduct simple scaling on the data [0,1]
sizeX= size(X,1);
Xnorm = (X - repmat(min(X),sizeX,1))./repmat(range(X),sizeX,1);
% Caution, remove uninformative NaN data % for nan - columns
Xcl = Xnorm(:,all(~isnan(Xnorm)));   

  
%% Naive Bayes 
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @predict_MNNB);
mean(accuracy)

%% Distance Metric
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'Distance','minkowski', 'Exponent',3, 'NumNeighbors',30),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);

%% Normalize 
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',30),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(Xcl, Y, 4, mdl);

%% KNN ~Various N 
acc = zeros(50,4);
for i = 1:50
    mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',i),testX);
    [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);
    acc(i,:) = accuracy;
end
plot acc;


%% K-means with 10 clusters
mdl2 = @(train_x,train_y,test_x,test_y) k_means(train_x,train_y,test_x,test_y, 10);
[accuracy, ~, ~] = cross_validation(Xcl, Y, 4, mdl2);

%% Kernel Regression

%%
folds = 4;
disp('linear regression + auto-encoder');
X = new_feat(1:n,:);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, folds, @linear_regression);
accuracy
mean(accuracy)

%%
addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @logistic);
accuracy
mean(accuracy)
%%
addpath('./liblinear');
addpath('./libsvm');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @kernel_libsvm);
accuracy
mean(accuracy)
