% Author: Max Lu
% Date: Nov 22


%% load data first ..


% Load the data first, see prepare_data.
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



%%


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





X = [words_train; words_test];
% X = normc(X);
Y = genders_train;
[n m] = size(words_train);

X = X(1:n, :);
bns = calc_bns(words_train,Y);
[top_bans, idx]=sort(bns,'descend');
word_sel=idx(1:1000);
X=X(:,word_sel);

Y = genders_train;

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