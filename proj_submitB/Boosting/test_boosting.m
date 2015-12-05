%% Test boosting
% Deng, Xiang
clear all
close all
load data\image_features_train.mat
load data\image_features_test.mat
load .\data\words_train.mat
load .\data\words_train_n.mat
load .\data\words_test.mat
load .\data\genders_train.mat
addpath('./mex');
%%
format
acc_ens=zeros(1,8);
Xfull=[words_train_n ,image_features_train];
%Xfull=round(Xfull);
Yfull=genders_train;

 
Yfull( Yfull==0 )=-1; %for boosting, convert 0 to -1
scale_bns=false;
[n, ~] = size(Xfull);
[parts] = make_xval_partition(n, 8);
for j=1:1
    
    row_sel1=(parts~=j);
    row_sel2=(parts==j);
    
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
    
    opt=[];
    M=100;
    [a, models]=boosting(@boost_nb_train,@boost_nb_predict,Xtrain, Y, M,opt);
    
    Yhats=zeros( size(Xtest,1),M);
    for i=1:M
        Yhats(:,i) = boost_nb_predict(models.mdl{i}, Xtest) ; % now it's 1 or -1
    end
    Yhat=sign(Yhats*a);    % final prediction from boosting
    acc_ens(j)=sum(Yhat==Ytest)/length(Ytest)
    confusionmat(Ytest,Yhat)
end
acc_ens
mean(acc_ens)
