%% To Try %% 
% -Look for trend in words data.. PCA -> cluster? Gaussian? 
% -K-means for image features 
% PCA on only the words data : wrap PCA in classifiers. 


        
%% playground
Y = genders_train;
X = words_train;
% boolean -> 
[coef, scores, latent] = pca(X);

%% plot word count and observe
mean_words_female = mean(X(logical(genders_train),:));
mean_words_male = mean(X(~logical(genders_train),:));

figure;
plot(1:5001, mean_words_female,'bo');
hold on
plot(1:5001, mean_words_male,'rx');

%% 
mean_words_diff = abs(mean_words_female - mean_words_male);
figure;
plot(1:5001, mean_words_diff);
[V, I] = sort(mean_words_diff,'descend' );

%% 76: 71.55% 
features_index = I(1:76)';
X_selected = X(:,features_index);
[accuracy, ~,~] = cross_validation(X_selected, Y, 4, mdl);
mean(accuracy)

%% max 16 neighbors, 76 words 72.89% 
accuS = zeros(100, 50);
% size(X_selected);
for i = 1:100
    features_index = I(1:i)';
    X_selected = X(:,features_index);
    for j = 1:50 
      mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',j),testX);
      [accuracy, ~,~] = cross_validation(X_selected, Y, 10, mdl);
      accu = mean(accuracy);
      accuS(i,j) = accu;
    end
end

%%



%% test Gaussian Mixture 
GMModel = fitgmdist(X,2,'Display','final', 'Start', 'plus');
threshold = [0.4,0.6];
P = posterior(GMModel,X);

n = size(X,1);
[~,order]= sort(P(:,1))

figure;
plot(1:n,P(order,1),'r-',1:n,P(order,2),'b-');
legend({'Cluster 1', 'Cluster 2'});
ylabel('Cluster Membership Score');
xlabel('Point Ranking');
title('GMM with Full Unshared Covariances');

%%
idx = cluster(gm,X);
idxBoth = find(P(:,1)>=threshold(1) & P(:,1)<=threshold(2));
numInBoth = numel(idxBoth)

figure;
gscatter(X(:,1),X(:,2),idx,'rb','+o',5);
hold on;
plot(X(idxBoth,1),X(idxBoth,2),'ko','MarkerSize',10);
legend({'Cluster 1','Cluster 2','Both Clusters'},'Location','SouthEast');
title('Scatter Plot - GMM with Full Unshared Covariances')
hold off;

%%
Xpca = scores(:,1:2);
figure;
plot(Xpca(logical(genders_train),2),'bo');
hold on
plot(Xpca(~logical(genders_train),2),'rx');
hold off
%Xpca(logical(genders_train),1),
%Xpca(~logical(genders_train),1),

%% X = words_train > 0;

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
B = TreeBagger(95,trainX,trainY, 'Method', 'classification');
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

%% 100 81.65%, 1000 85.79% 1500 86.01% 1800 86.15% 3000 85.97% 4000 86.47% 4500 86.53% 5000 86.96%
addpath('./liblinear');
features_index = I(1:4000)';
X_selected = X(:,features_index);
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X_selected, Y, 4, @logistic);
accuracy
mean(accuracy)
%%
addpath('./liblinear');
addpath('./libsvm');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @kernel_libsvm);
accuracy
mean(accuracy)
