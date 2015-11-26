% SVM on words
% Deng, Xiang
% 11/26/2015
clear all
close all
load data\image_features_train.mat
load data\image_features_test.mat
load data\genders_train.mat
load data\words_train.mat
addpath('./liblinear/proj');


%%
Xtrainset=[words_train];
Y=genders_train;

[n, ~] = size(Xtrainset);
[parts] = make_xval_partition(n, 8);
clc
acc_ens=zeros(8,1);


bns = calc_bns(Xtrainset,Y,0.01);
IG=calc_information_gain(Y,Xtrainset,[1:size(Xtrainset,2)],10);
[top_igs, idx_ig]=sort(IG,'descend');
[top_bns, idx_bns]=sort(bns,'descend');


% Trains a SVM using liblinear and evaluates on test data.
% type : set type of solver (default 1)
%   for multi-class classification
% 	 1 -- L2-regularized L2-loss support vector classification (dual)
% 	 3 -- L2-regularized L1-loss support vector classification (dual)
% 	 5 -- L1-regularized L2-loss support vector classification
%    6 -- L1-regularized logistic regression
% 	 7 -- L2-regularized logistic regression (dual)
% 	12 -- L2-regularized L2-loss support vector regression (dual)
% 	13 -- L2-regularized L1-loss support vector regression (dual)
% Xtrain - train data
% Ytrain - train label
% s      - type for -s option
% c      - cost parameter
s=6;
c=0.1;
option  = sprintf('-s %d -q -c %g', s, c);
%words_train_s=bsxfun(@times,words_train,IG);
for i=1:8
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    
    %cols_sel=idx_ig(1:1000);
    cols_sel=idx_bns(1:2000);
    %cols_sel=unique([idx_ig(1:1000),idx_bns(1:2000)]); % or ensemble the
    %top features from both ig and bns
    Xtrain=Xtrainset(row_sel1,cols_sel);
    Ytrain=Y(row_sel1);
    Xtest=Xtrainset(row_sel2,cols_sel);
    Ytest=Y(row_sel2);
    

    model   = liblinear_train(Ytrain,sparse( Xtrain), option); 
    Yhat = liblinear_predict(ones(size(Xtest,1),1),sparse(Xtest),model, '-b 1 -q');
    acc_ens(i)=sum(round(Yhat)==Ytest)/length(Ytest)
    confusionmat(Ytest,Yhat)

end
acc_ens
mean(acc_ens)