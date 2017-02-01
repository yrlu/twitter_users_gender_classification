%Deng, Xiang
% 11 20-
clear all
close all
load data\words_train.mat;
load ('data\genders_train.mat');
occupany= sum(sum(words_train~=0))/(4998*5000)
% This is NOT a sparse matrix!!!

%% run PCA recontruction
X_full=words_train;
X_mean=mean(X_full);
%A = [1 2 10; 1 4 20;1 6 15] ;
%C = bsxfun(@minus, A, mean(A))
% above is deviation from mean example
[coeff, score, latent]=pca(X_full);
%%
accuracy=zeros(size(coeff,1),1);
X_dev=bsxfun(@minus, X_full, mean(X_full));
for i=300:330
    Xp=score(:,1:i)*coeff(:,1:i)';
    err_orig=norm(X_dev,'fro');
    Xp_dev=Xp-X_dev; %(X-(Xp+mean(X))=(Xdev-Xp)
    err=norm(Xp_dev,'fro');
    accuracy(i)=1-(err^2/err_orig^2);
end
%
figure
plot(1:size(coeff,1),accuracy)
title ('recontruction accuracy vs number of principle components')
xlabel('PC#')
ylabel('Accuracy')
hold on
grid on

%%
male_idx=find(genders_train==0);
female_idx=find(genders_train==1);
X_male_train=words_train(male_idx,:);
X_female_train=words_train(female_idx,:);
[coeff_male, score_male, latent_male]=pca(X_male_train);
[coeff_female, score_female, latent_female]=pca(X_female_train);

%% Naive bayes
Nc=3000;
cols_sel=[1:320];
X_train_split_train=score(1:Nc,cols_sel);
X_train_split_train_labels=genders_train(1:Nc);
X_train_split_test=score(Nc+1:end,cols_sel);
X_train_split_test_labels=genders_train(Nc+1:end);
nb_train = NaiveBayes.fit(X_train_split_train , X_train_split_train_labels);
cpre = nb_train.predict(X_train_split_test);
err=sum(X_train_split_test_labels ~= cpre)/size(X_train_split_test_labels,1);%compute error
accuracy_orig=1-err

%% logistic
addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(score(:,1:2000), genders_train, 8, @logistic);
accuracy
mean(accuracy)
% Nc=3000;
% cols_sel=[1:1000];
% X_train_split_train=score(1:Nc,cols_sel);
% X_train_split_test=score(Nc+1:end,cols_sel);
% X_train_split_train_labels=genders_train(1:Nc);
% X_train_split_test_labels=genders_train(Nc+1:end);
% [  predicted_label ] = logistic( X_train_split_train, X_train_split_train_labels,X_train_split_test, X_train_split_test_labels );
%  precision_ori_log=1-sum(X_train_split_test_labels ~= predicted_label)/size(predicted_label,1)

%% Look at feature intersections
male_idx=find(genders_train==0);
female_idx=find(genders_train==1);
X_male_train=words_train(male_idx,:);
X_female_train=words_train(female_idx,:);
mean_train_male=mean(X_male_train);
bin_X_male_train=X_male_train;
% for i=1:7
%     bin_X_male_train(:,i)=bin_X_male_train(:,i)/ mean_train_male(i);
% end
Inter_X_male_train=bin_X_male_train'*bin_X_male_train;
%Inter_X_male_train=Inter_X_male_train/det(Inter_X_male_train);
figure
imagesc(Inter_X_male_train);
%colormap('gray')

mean_train_female=mean(X_female_train);
bin_X_female_train=X_female_train;
for i=1:7
    bin_X_female_train(:,i)=bin_X_female_train(:,i)/ mean_train_female(i);
end
Inter_X_female_train=bin_X_female_train'*bin_X_female_train;
%Inter_X_female_train=Inter_X_female_train/det(Inter_X_female_train);
figure
imagesc(Inter_X_female_train);
%colormap('gray')

%% Feature selection

a=3000;
b=3100;
row_sel1=[a:b];
row_sel2=[1:a-1,b+1:size(words_train,1)];
cols_sel=[1:5000];
Xtrain=words_train(row_sel1,cols_sel);
Ytrain=genders_train(row_sel1);
Xtest=words_train(row_sel2,cols_sel);
Ytest=genders_train(row_sel2);

bns = calc_bns(Xtrain,Ytrain);
