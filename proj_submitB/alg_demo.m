% Author: Dongni Wang
% Date: Dec 3, 2015
%
% This file demonstrates the prediction results of our four algorithms.
%

%% Add path
addpath('./NB');
addpath('./KNN');
addpath('./PCA');
addpath('./liblinear');
addpath('./liblinear/proj');
addpath('./libsvm');
addpath('./Boosting');
addpath('./Bagging');
addpath('./mex');
addpath('./NBfast');
addpath('./Feature');
%% Prepare data
run prepare_data.m
%% Load data
% run prepare_data.m for data process (.txt to .mat)
load('../train/genders_train.mat', 'genders_train');
load('../train/images_train.mat', 'images_train');
load('../train/image_features_train.mat', 'image_features_train');
load('../train/words_train.mat', 'words_train');
load('../train/words_train_n.mat', 'words_train_n');
load('../test/images_test.mat', 'images_test');
load('../test/image_features_test.mat', 'image_features_test');
load('../test/words_test.mat', 'words_test');
load('../test/words_test_n.mat', 'words_test_n');
%% Variables
Xtrain = words_train;
Ytrain = genders_train;
Xtest = words_test;
Ytest = ones(size(words_test,1),1);

%% Naive Bayes
[Yhat, ~] = predict_MNNB(Xtrain, Ytrain, Xtest, Ytest);

%% Logistic Regression
Yhat = logistic(Xtrain, Ytrain, Xtest, Ytest);

%% ANN
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
Xtrain = [words_train; words_train(1,:); words_train(2,:)];
Ytrain = [genders_train; genders_train(1); genders_train(2)];
[Yhat,~] = acc_neural_net(Xtrain, Ytrain, Xtest, Ytest);

%% LogitBoost + Trees
IG=calc_information_gain(Ytrain,Xtrain,[1:5000],10);
[~, idx]=sort(IG,'descend');
word_sel=idx(1:350);
Xtrain_selected =Xtrain(:,word_sel);
Xtest_selected = Xtest(:,word_sel);
Yhat = acc_ensemble_trees(Xtrain_selected, Ytrain, Xtest_selected, Ytest);

%% K-nearest Neighbors
IG=calc_information_gain(Ytrain,Xtrain,[1:5000],10);
[~, idx]=sort(IG,'descend');
word_sel=idx(1:70);
Xtrain_selected =Xtrain(:,word_sel);
Xtest_selected = Xtest(:,word_sel);

Yhat = knn_test(16, Xtrain_selected, Ytrain, Xtest_selected);

%% Please go to autoencoder.m to test the performance of autoencoder. 

%% PCA  (This one is not what we used in our classifier. Please refer to README.txt,
% image_features_extract.m and next section for better test results).
[~, ~, ~, train_grey] = convert_to_img(images_train);
[~, ~, ~, test_grey] = convert_to_img(images_test);
X = cat(3, train_grey, test_grey);
[h w n] = size(X);
x = reshape(X,[h*w n])

[U mu vars] = pca_toolbox(x);
[YPC,~,~] = pcaApply(x, U, mu, 2000 );
YPC = double(YPC');

Xtrain_pca = YPC(1:size(train_grey,3),:);
Xtest_pca = YPC(size(train_grey,3)+1:end,:);
[Yhat,~] = svm_predict(Xtrain_pca, Ytrain, Xtest_pca, Ytest);

%% PCA (on features generated by test_face_detection.m)
% Note: this part is only tested on face-detected images, thus the size of
% the testing results is smaller than the test data size.
% For the train_hog, we used the first 2 observations twice to make partition of
% cross-validation easier. This seems to have little impact on the
% classifier.
image_features_extract
n_train = sum(certain(1:5000,:));
PC_train = YPC(1:n_train,:);
PC_test = YPC(n_train+1:end,:);
Ytest = ones(size(PC_test,1),1);
[Yhat,~] = svm_predict(PC_train, train_y_certain, PC_test, Ytest);

%% Ensemble methods: Adaboost(our own implementation)
% please get into the mex folder and run make_mex.m 
acc_ens=zeros(1,9);
Xfull=[words_train_n ,image_features_train];
Yfull=genders_train;
Yfull( Yfull==0 )=-1; %for boosting, convert 0 to -1
[n, ~] = size(Xfull);
[parts] = make_xval_partition(n, 8);
row_sel1=(parts~=1);
row_sel2=(parts==1);

Xtrain=Xfull(row_sel1,:);
Y=Yfull(row_sel1,:);
Xtest=Xfull(row_sel2,:);
Ytest=Yfull(row_sel2);

bns = calc_bns(Xtrain,Y,0.05);
[top_bns, idx]=sort(bns,'descend');
word_sel=idx(1:600);

Xtrain=Xtrain(:,word_sel);
Xtrain=bsxfun(@times,Xtrain,bns(word_sel) );%------scale the columns by bns_i s
Xtrain=round(Xtrain);

Xtest=Xtest(:,word_sel);
Xtest=bsxfun(@times,Xtest,bns(word_sel) );%------scale the columns by bns_i s
Xtest=round(Xtest);
cnt=1;
for t=20:10:100
    opt=[];
    
    M=t;
    [a, models]=boosting(@boost_nb_train,@boost_nb_predict,Xtrain, Y, M,opt);
    
    Yhats=zeros( size(Xtest,1),M);
    for i=1:M
        Yhats(:,i) = boost_nb_predict(models.mdl{i}, Xtest) ; % now it's 1 or -1
    end
    Yhat=sign(Yhats*a);    % final prediction from boosting
    acc_ens(cnt)=sum(Yhat==Ytest)/length(Ytest);
    cnt=cnt+1;
    %confusionmat(Ytest,Yhat)
end
acc_ens
mean(acc_ens)
figure
plot(20:10:100,acc_ens)
xlabel('number of models')
ylabel('accuracy')
title('Adaboost, number of models(NB) vs. accuracy')
%% Ensembling methods: Bagging
Xfull=[words_train,image_features_train];
Yfull=genders_train;
scale_bns=0;
[n, ~] = size(Xfull);
[parts] = make_xval_partition(n, 8);
% Bagging with 15 models, test accuracy
for j=1:8
    
    row_sel1=(parts~=j);
    row_sel2=(parts==j);
    
    Xtrain=Xfull(row_sel1,:);
    Y=Yfull(row_sel1,:);
    Xtest=Xfull(row_sel2,:);
    Ytest=Yfull(row_sel2);
    
    s=6;
    c=0.15;
    F=0.6;% fraction of data for training each model
    M=15; % number of linear models
    numfeat=1000;%number of features
    [models_linear,cols_sel]=train_bag_linear(Xtrain,Y,numfeat,0,scale_bns,s,c,F,M);
    % test
    Xtest_cur=Xtest(:,cols_sel );
    YhatA=predict_bagged_linear(models_linear,Xtest_cur,M);
    %Yhat=YhatsT(:,1)
    acc_ens(j)=sum(YhatA==Ytest)/length(Ytest);
    confusionmat(Ytest,YhatA);
end
acc_ens
mean(acc_ens)
% in comparison, no bagging and test accuracy
for j=1:8
    
    row_sel1=(parts~=j);
    row_sel2=(parts==j);
    
    Xtrain=Xfull(row_sel1,:);
    Y=Yfull(row_sel1,:);
    Xtest=Xfull(row_sel2,:);
    Ytest=Yfull(row_sel2);
    
    s=6;
    c=0.15;
    F=1;% fraction of data for training each model
    M=1; % number of linear models
    numfeat=1000;%number of features
    [models_linear,cols_sel]=train_bag_linear(Xtrain,Y,numfeat,0,scale_bns,s,c,F,M);
    % test
    Xtest_cur=Xtest(:,cols_sel );
    YhatA=predict_bagged_linear(models_linear,Xtest_cur,M);
    %Yhat=YhatsT(:,1)
    acc_ens(j)=sum(YhatA==Ytest)/length(Ytest);
    confusionmat(Ytest,YhatA);
end
acc_ens
mean(acc_ens)