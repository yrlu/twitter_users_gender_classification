%% The playground to mess around... 

% -Look for trend in words data.. PCA -> cluster? Gaussian?
% bns = calc_bns(words_train,Y); %-------------feature selection opt1% 
% IG=calc_information_gain(genders_train,words_train,[1:5000],10); %trees

% -K-means for image features 
% PCA on only the words data : wrap PCA in classifiers. 
% Decision Tree 
% GMM --- not very effective at this stage (on raw data)
% Image features
% tf-idf

% 

%%
[accu, ~,~]= cross_validation_idx(5000, 5, @add_classifier_test);
mean(accu);



%%
% Bernoulli: 5000: 78.99% 2000: 79.71%; 1000: 80.39% 500: 81.19%  
% 450: 81.53% 400: 81.65% 355: 81.87% 350: 81.73% 300: 81.31% 200: 80.45%
% 
% words_train_s = [words_train, image_features_train];
% words_train_s = [words_train_s; words_train_s(1,:); words_train_s(2,:)];
% genders_train_s = [genders_train; genders_train(1);genders_train(2)];
X =[words_train, image_features_train];
IG=calc_information_gain(genders_train,[words_train, image_features_train],[1:size([words_train, image_features_train],2)],10);
[top_igs, index]=sort(IG,'descend');

cols_sel=index(1:355);
% prepare data for ensemble trees to train and test.
train_x_fs = X(:, cols_sel);

%
[accu, ~,~]= cross_validation(train_x_fs,Y, 5, @predict_MNNB);
mean(accu)


%%
% 3 + knn + V-NB: 89.02
% 3: 88.76%
% 3 + B-NB (IG350): 0.8934/ 89.48/ 89.6

        
%% playground
Y = genders_train;
X = words_train;
% [coef, scores, latent] = pca(X);

%% Test ensemble methods % 500: 87.77% ; 1000: 88.59%; 1500: 88.14%
[n, ~] = size(words_train);
[parts] = make_xval_partition(n, 8);
clc
acc=zeros(8,1);
acc_nb=zeros(8,1);
acc_log=zeros(8,1);
bns = calc_bns(words_train,Y);

new_features = [words_train,image_features_train];
IG=calc_information_gain(genders_train,new_features, [1:size(new_features,2)],10);

%words_train_s=bsxfun(@times,words_train,bns);
% words_train_s=bsxfun(@times,new_features,IG);
[top_igs, idx]=sort(IG,'descend');
%[top_bans, idx]=sort(bns,'descend'); 84.6% ~ 1000 
words_train_s=new_features;% words_train;
for i=1:8
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:1000);
    
    Xtrain=words_train_s(row_sel1,cols_sel);
    Ytrain=genders_train(row_sel1);
    Xtest=words_train_s(row_sel2,cols_sel);
    Ytest=genders_train(row_sel2);
    
    %templ = templateTree('MaxNumSplits',1);
    %ens = fitensemble(Xtrain,Ytrain,'GentleBoost',200,'Tree');
    %ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree');
    ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree' ); %ens=regularize(ens);
    %ens = fitensemble(Xtrain,Ytrain, 'RobustBoost',300,'Tree','RobustErrorGoal',0.01,'RobustMaxMargin',1);
    %
    Yhat= predict(ens,Xtest);
    acc_ens(i)=sum(round(Yhat)==Ytest)/length(Ytest);
    confusionmat(Ytest,Yhat)
    figure;
    plot(loss(ens,Xtest,Ytest,'mode','cumulative'));
    xlabel('Number of trees');
    ylabel('Test classification error');
end
acc_ens
 mean(acc_ens)


%% plot features count and observe
mean_words_female = mean(X(logical(genders_train),:));
mean_words_male = mean(X(~logical(genders_train),:));

mean_image_features_f = mean(image_features_train(logical(genders_train),:));
mean_image_features_m = mean(image_features_train(~logical(genders_train),:));

var_image_features_f = var(image_features_train(logical(genders_train),:));
var_image_features_m = var(image_features_train(~logical(genders_train),:));
%%
figure;
plot(1:7, var_image_features_f,'bo');
hold on
plot(1:7, var_image_features_m,'rx');

%% 
mean_words_diff = abs(mean_words_female - mean_words_male);
figure;
plot(1:5001, mean_words_diff);
[V, I] = sort(mean_words_diff,'descend' );

%%
close all
 bns = calc_bns(words_train,Y); %-------------feature selection opt1% 
 IG=calc_information_gain(genders_train,words_train,[1:5000],10);% --------or try to compute the information gain
 
 % you can further scale the words
 %words_train_s=bsxfun(@times,words_train,bns);% or...
 %words_train_s=bsxfun(@times,words_train,IG);

 [top_igs, idx]=sort(IG,'descend'); %---- and sort

% % and pick the top words
 word_sel=idx(1:1000);
 X_selected =X(:,word_sel);




%% 
 
 %% 76: 71.55% knn
features_index = I(1:76)';
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',j)
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
Xpca = scores(:,1:2);
figure;
plot(Xpca(logical(genders_train),2),'bo');
hold on
plot(Xpca(~logical(genders_train),2),'rx');
hold off

%% X = words_train > 0;

%figure, plot(cumsum(latent)/sum(latent));
plot(cumsum(latent)/sum(latent));
xlabel('Number of Principal Components');
ylabel('Reconstruction accuracy');
% Reconstruction PCA
% 90%: 25; 95%: 84; 99%: 512
numpc = find(cumsum(latent)/sum(latent)>=0.9,1);


%% 71.31+% minkowski; 72.13% euclidean
% X = scores(:,1:25);
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',30),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);
mean(accuracy)

%%  

%% fount 12 neighbors work the best. 
X = scores(:,1:25);
accu = zeros(30,1);
for i = 1:50
    mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',i),testX);
    [accuracy, ~,~] = cross_validation(X, Y, 4, mdl);
    accu(i) = mean(accuracy);
end

plot(accu)

%% test majority vote
X = [words_train, image_features_train];
Y = genders_train;
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @majority_vote);
mean(accuracy)

%% 71.43%
X = scores(:,1:84);
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'Distance','minkowski', 'Exponent',3, 'NumNeighbors',30),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);
mean(accuracy)



%% Random Forest packed
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @random_forest);



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
B = fitctree(nTrees,X,Y, 'Method', 'classification', 'OOBPred','On');


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
  
%% MN Naive Bayes 
% Vanilla 63.59%  boolean 79%
% 2000: V 67.57% 78.85% Boolean 2*79.47% 
% After 1000 words selection: Vanilla 69.91% 80.09% Boolean 80.39% bns 79.39%
% 500: Vanilla 72.27% 79.27% B 81.25% b 79.77% 
% 300: V 75.61% 80.51% B 81.35% 81.35% b 80.81% 
% 100: V 77.95% 75.39% B in 74.67% b 74.85%
% 50: V 69.55% bns 69.13% 66.15% 66.75%

% ~ boolean selected words: 400 81.75% 

%% Boolean ~IG 350 81.79%


Y = genders_train;
X = [words_train, image_features_train];
IG=calc_information_gain(genders_train,X, [1:size(X,2)],10);
[~, idx]=sort(IG,'descend'); %---- and sort

% % and pick the top words
word_sel=idx(1:50);
X =X(:,word_sel);

%%
[n, ~] = size(words_train);
[parts] = make_xval_partition(n, 8);
clc
acc=zeros(8,1);
acc_nb=zeros(8,1);
acc_log=zeros(8,1);
bns = calc_bns(words_train,Y);

new_features = [words_train,image_features_train];
 IG=calc_information_gain(genders_train,new_features, [1:size(new_features,2)],10);
%IG = calc_bns(words_train,Y);

%words_train_s=bsxfun(@times,words_train,bns);
% words_train_s=bsxfun(@times,new_features,IG);
[top_igs, idx]=sort(IG,'descend');
%[top_bans, idx]=sort(bns,'descend'); 84.6% ~ 1000 
words_train_s=new_features;% words_train;
for i=1:8
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:370);
    
    Xtrain=words_train_s(row_sel1,cols_sel);
    Ytrain=genders_train(row_sel1);
    Xtest=words_train_s(row_sel2,cols_sel);
    Ytest=genders_train(row_sel2);
    
    [Yhat, ~] = predict_MNNB(Xtrain, Ytrain, Xtest, Ytest);
   
    acc_ens(i)=sum(round(Yhat)==Ytest)/length(Ytest);
    confusionmat(Ytest,double(Yhat))
   
end
acc_ens
 mean(acc_ens)

%words_train_s=bsxfun(@times,words_train,bns);
% words_train_s=bsxfun(@times,new_features,IG);

%[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 8, @predict_MNNB);
%mean(accuracy)

%%
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @NB);
mean(accuracy)


%% IG selected KNN feature, 0.7335
mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',12),testX);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);
mean(accuracy)
%% KNN ~Various N 
acc = zeros(50,4);
for i = 1:50
    mdl = @(trainX, trainY, testX, testY) predict(fitcknn(trainX,trainY, 'NumNeighbors',i),testX);
    [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, mdl);
    acc(i,:) = accuracy;
end
plot acc;


%% K-means with 10 clusters k-means

mdl2 = @(train_x,train_y,test_x,test_y) k_means(train_x,train_y,test_x,test_y, 10);
[accuracy, ~, ~] = cross_validation(X, Y, 4, mdl2);
mean(accuracy)
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
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @logistic);
accuracy
mean(accuracy)
%%
addpath('./liblinear');
addpath('./libsvm');
disp('svm + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @kernel_libsvm);
accuracy
mean(accuracy)


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

