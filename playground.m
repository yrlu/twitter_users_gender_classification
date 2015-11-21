%% To Try %% 
% -Look for trend in words data.. PCA -> cluster? Gaussian? 
% -K-means for image features 

% PCA on only the words data
% wrap PCA in classifiers. 

        
%% playground
Y = genders_train;
X = words_train;
% boolean -> 
%% X = words_train > 0;
[coef, scores, latent] = pca(X);

%figure, plot(cumsum(latent)/sum(latent));
plot(cumsum(latent)/sum(latent));
xlabel('Number of Principal Components');
ylabel('Reconstruction accuracy');
% Set you numpc here
% 90%: 25; 95%: 84; 99%: 512
numpc = find(cumsum(latent)/sum(latent)>=0.9,1);


%% 71.31+% minkowski; 72.13% euclidean
X = scores(:,1:25);
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',30),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);
mean(accuracy)

%% fount 12 neighbors work the best. 
X = scores(:,1:25);
accu = zeros(30,1);
for i = 1:50
    mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',i),testX);
    [accuracy, ~,~] = cross_validation(X, Y, 4, mdl);
    accu(i) = mean(accuracy);
end

plot(accu)

%% 71.43%
X = scores(:,1:84);
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'Distance','minkowski', 'Exponent',3, 'NumNeighbors',30),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);
mean(accuracy)


%% Random Forest packed
B = TreeBagger(95,trainX,trainY, 'Method', 'classification', 'OOBPred','On');
RFpredict = @(test_x) sign(str2double(B.predict(test_x)) - 0.5);

%% data loading:
%% Load the data first, see data_preprocess.m

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




%% normalization
% X = [words_train, image_features_train; words_test, image_features_test];

% Conduct simple scaling on the data [0,1]
sizeX= size(X,1);
Xnorm = (X - repmat(min(X),sizeX,1))./repmat(range(X),sizeX,1);
% Caution, remove uninformative NaN data % for nan - columns
Xcl = Xnorm(:,all(~isnan(Xnorm)));   

%% Random Forest ~ treeBagger
rng default
nTrees = 20; % # of trees
% Decision Forest
B = TreeBagger(nTrees,X,Y, 'Method', 'classification', 'OOBPred','On');


oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%%
%% 
Xtrain = X;
trainX = Xtrain(1:4000,:);
trainY = Y(1:4000,:);
testX = Xtrain(4001:end,:);
testY = Y(4001:end,:);
%% ==>  90- 82.79% 95-83.21%, 100-82.99% 125-82.63%
[accuracy, ~, ~] = cross_validation(X, Y, 4, @random_forest);
mean(accuracy)
  
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
