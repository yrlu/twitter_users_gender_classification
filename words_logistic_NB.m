%% READ ME
% Deng, Xiang 11/22/2015

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
%%
clear all
close all
load .\data\words_train.mat
load .\data\words_test.mat
load .\data\genders_train.mat
tic
% X = [words_train, image_features_train; words_test, image_features_test];
X = [words_train; words_test];
% X = normc(X);
Y = genders_train;
[n m] = size(words_train);

% PCA features
% X = scores(1:n, 1:3200);
X = X(1:n, :);
bns = calc_bns(words_train,Y);
[top_bans, idx]=sort(bns,'descend');
word_sel=idx(1:1000);
X=X(:,word_sel);

addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 8, @logistic);
accuracy
max(accuracy)
toc

%% NB fit
clear all
close all
load .\data\words_train.mat
load .\data\words_test.mat
load .\data\genders_train.mat
tic
% X = [words_train, image_features_train; words_test, image_features_test];
X = [words_train; words_test];
% X = normc(X);
Y = genders_train;
[n m] = size(words_train);

% PCA features
% X = scores(1:n, 1:3200);
X = X(1:n, :);
bns = calc_bns(words_train,Y);
[top_bans, idx]=sort(bns,'descend');
word_sel=idx(1:300);
X=X(:,word_sel);

addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 8, @NB);
accuracy
max(accuracy)
toc

%% The max possible accuracy by combining NB and logistic
[n, ~] = size(words_train);
[parts] = make_xval_partition(n, 8);
clc
acc=zeros(8,1);
acc_nb=zeros(8,1);
acc_log=zeros(8,1);
bns = calc_bns(words_train,Y);

IG=calc_information_gain(genders_train,words_train,[1:5000],10);

%words_train_s=bsxfun(@times,words_train,bns);
words_train_s=bsxfun(@times,words_train,IG);
[top_igs, idx]=sort(IG,'descend');
%[top_bans, idx]=sort(bns,'descend');
%words_train_s=words_train;
for i=1:8
    
    
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    
    cols_sel=idx(1:500);
    %cols_sel=idx(1:850);
    
    % mean_img=mean(b_image_features_train);
    % for i=1:7
    %     b_image_features_train(:,i)=b_image_features_train(:,i)./mean_img(i);
    % end
    Xtrain=words_train_s(row_sel1,cols_sel);
    Ytrain=genders_train(row_sel1);
    Xtest=words_train_s(row_sel2,cols_sel);
    Ytest=genders_train(row_sel2);
    
    NB_pred=NB(Xtrain,Ytrain,Xtest,Ytest);
    Log_pred=logistic(Xtrain,Ytrain,Xtest,Ytest);
    good_pred_ensemble=1-(NB_pred~=Ytest).*(Log_pred~=Ytest);
    
    acc(i)=sum(good_pred_ensemble)/length(Ytest);
    acc_nb(i)=sum(NB_pred==Ytest)/length(Ytest);
    acc_log(i)=sum(Log_pred==Ytest)/length(Ytest);
end
disp('The max possible accuracy by combining NB and logistic');
acc'
disp('accuracy of NB');
acc_nb'
disp('accuracy of log');
acc_log'

%% Test ensemble methods
close all
[n, ~] = size(words_train);
[parts] = make_xval_partition(n, 8);
clc
acc_ens=zeros(8,1);


bns = calc_bns(words_train,Y);
IG=calc_information_gain(genders_train,words_train,[1:5000],10);
[top_bans, idx]=sort(IG,'descend');
%words_train_s=bsxfun(@times,words_train,IG);
for i=1:8
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:500);
    
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
%%

train_x = words_train;
train_y = genders_train;
test_x = words_test;
test_y = ones(size(test_x,1), 1);

model = train(train_y, sparse(train_x), ['-s 0', 'col']);
[Yhat] = predict(test_y, sparse(test_x), model, ['-q', 'col']);
% dlmwrite('submit.txt',Yhat, '\n');