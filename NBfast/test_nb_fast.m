% Deng, Xiang 11/28/2015
%% NB fit fast
clear all
close all
load data\image_features_train.mat
load data\image_features_test.mat
load .\data\words_train.mat
load .\data\words_train_n.mat
load .\data\words_test.mat
load .\data\genders_train.mat
addpath('./mex');
%% Naive bayes requires the normalization of data, not necessary for logistic or trees
format
acc_ens=zeros(1,8);
Xfull=[words_train_n ,image_features_train];
%Xfull=[words_train ,image_features_train];
%Xfull=round(Xfull);
Yfull=genders_train;
scale_bns=false;
[n, ~] = size(Xfull);
[parts] = make_xval_partition(n, 8);
for j=1:8
    
    row_sel1=(parts~=j);
    row_sel2=(parts==j);
    
    Xtrain=Xfull(row_sel1,:);
    Y=Yfull(row_sel1,:);
    Xtest=Xfull(row_sel2,:);
    Ytest=Yfull(row_sel2);
    
    bns = calc_bns(Xtrain,Y,0.05);
    bns=bns/max(bns);
    [top_bns, idx]=sort(bns,'descend');
    word_sel=idx(1:600);
    
    Xtrain=Xtrain(:,word_sel);
    Xtrain=bsxfun(@times,Xtrain,bns(word_sel) );%------scale the columns by bns_i s
    Xtrain=round(Xtrain);
    
    Xtest=Xtest(:,word_sel);
    Xtest=bsxfun(@times,Xtest,bns(word_sel) );%------scale the columns by bns_i s
    Xtest=round(Xtest);
    
    % model = train_fastnb(sparse(Xtrain), Y, [0 1]);
    
    F=1; % fraction of subset, 0 to 1
    M=1;
    models=train_bag_nb_fast(Xtrain,Y,F,M);
    Yhat=predict_bagged_nb_fast(models,Xtest,M);
%     Yhats=zeros( size(Xtest,1),M);
%     
%     % see how does each classifier predict
%     for i=1:M
%         P_priors = predict_fastnb(models.NB{i}, sparse(Xtest)');
%         [~, Yhat] = max(P_priors, [], 2);
%         Yhats(:,i)=Yhat-1;
%     end
%     YYY=[Yhats,Ytest];
%     sss=sum(Yhats,2)/M;
%     amb_ind=find(sss(find(sss>0))<1);
    
%     potential= (length(amb_ind)-sum(mode(Yhats(amb_ind),2)==Ytest(amb_ind)))/length(sss) %-----the ambiguous portion
%     Yhat=round(mode(Yhats,2));
    
    % average para from each classifier, this is almost equivalent to mode
    % the predictions
    %     est_paras=zeros(size(models.NB{1}.feature_prob));
    %     est_prior=zeros(size(models.NB{1}.class_prob));
    %     for i=1:M
    %         est_paras=est_paras+exp(models.NB{i}.feature_prob);
    %         est_prior=est_prior+exp(models.NB{i}.class_prob);
    %     end
    %     est_paras=log(est_paras./M);
    %     est_prior=log(est_prior./M);
    %     model_est=models.NB{1};%copy the datasetructure and replace paras
    %     model_est.feature_prob=est_paras;
    %     model_est.class_prob=est_prior;
    %     P_priors = predict_fastnb(model_est, sparse(Xtest)');
    %     [~, Yhat] = max(P_priors, [], 2);
    %     Yhat(find(Yhat==1))=0;
    %     Yhat(find(Yhat==2))= 1;
    %
    acc_ens(j)=sum(Yhat==Ytest)/length(Ytest)
    confusionmat(Ytest,Yhat)
end
acc_ens
mean(acc_ens)
