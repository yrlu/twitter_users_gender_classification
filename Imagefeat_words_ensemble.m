% Deng, Xiang 
% 11/22/2015
% 87.53%....
%% READ ME 
%% Note IG or BNS captures different sets of words, using which one depends on the classifier, by experiments IG works well for stump trees
% bns = calc_bns(words_train,Y); %-------------feature selection opt1% 
% IG=calc_information_gain(genders_train,words_train,[1:5000],10);% --------or try to compute the information gain
% 
% % you can further scale the words
% %words_train_s=bsxfun(@times,words_train,bns);% or...
% words_train_s=bsxfun(@times,words_train,IG);

% [top_igs, idx]=sort(IG,'descend'); %---- and sort

% % and pick the top words
% word_sel=idx(1:300);
% X=X(:,word_sel);
%% ensemble image feature + words + select the top features
clear all
close all
load data\image_features_train.mat
load data\image_features_test.mat
load data\genders_train.mat
load data\words_train.mat
Xtrainset=[words_train,image_features_train];
Y=genders_train;

[n, ~] = size(Xtrainset);
[parts] = make_xval_partition(n, 8);
clc
acc_ens=zeros(8,1);


bns = calc_bns(Xtrainset,Y,0.01);
IG=calc_information_gain(Y,Xtrainset,[1:size(Xtrainset,2)],10);
[top_igs, idx]=sort(IG,'descend');
%[top_bns, idx]=sort(bns,'descend');
%words_train_s=bsxfun(@times,words_train,IG);
for i=1:8
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:400);
    
    Xtrain=Xtrainset(row_sel1,cols_sel);
    Ytrain=Y(row_sel1);
    Xtest=Xtrainset(row_sel2,cols_sel);
    Ytest=Y(row_sel2);
    
    %templ = templateTree('MaxNumSplits',1);
    %ens = fitensemble(Xtrain,Ytrain,'GentleBoost',200,'Tree');
    %ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree');
    ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree' ); %ens=regularize(ens);
    %ens = fitensemble(Xtrain,Ytrain, 'RobustBoost',300,'Tree','RobustErrorGoal',0.01,'RobustMaxMargin',1);
    %
    Yhat= predict(ens,Xtest);
    acc_ens(i)=sum(round(Yhat)==Ytest)/length(Ytest)
    confusionmat(Ytest,Yhat)
    figure;
    plot(loss(ens,Xtest,Ytest,'mode','cumulative'));
    xlabel('Number of trees');
    ylabel('Test classification error');
end
acc_ens
mean(acc_ens)
%% Predict
clear all
close all
load data\image_features_train.mat
load data\image_features_test.mat
load data\genders_train.mat
load data\words_train.mat
load data\words_test.mat
Xtrainset=[words_train,image_features_train];
Ytrain=genders_train;
Xtest=[words_test,image_features_test];

clc
bns = calc_bns(Xtrainset,Ytrain);
IG=calc_information_gain(Ytrain,Xtrainset,[1:size(Xtrainset,2)],10);
[top_igs, idx]=sort(IG,'descend');
cols_sel=idx(1:400);

Xtrain=Xtrainset(:,cols_sel);
Xtest=Xtest(:,cols_sel);
ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree' );
Yhat= predict(ens,Xtest);
dlmwrite('submit2.txt',Yhat, '\n');