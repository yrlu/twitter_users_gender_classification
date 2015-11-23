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

folds = 5;

words_train_s = [words_train, image_features_train];
words_train_s = [words_train_s; words_train_s(1,:); words_train_s(2,:)];
genders_train_s = [genders_train; genders_train(1);genders_train(2)];

[n, ~] = size(words_train_s);
[parts] = make_xval_partition(n, folds);
clc
acc_ens = zeros(folds, 1);
acc=zeros(folds,1);
acc_nb=zeros(folds,1);
acc_log=zeros(folds,1);
% bns = calc_bns([words_train, image_features_train],genders_train_s);

IG=calc_information_gain(genders_train,[words_train, image_features_train],[1:size([words_train, image_features_train],2)],10);

%words_train_s=bsxfun(@times,words_train,bns);
% words_train_s=bsxfun(@times,[words_train, image_features_train],IG);

[top_igs, idx]=sort(IG,'descend');

for i=1:folds
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:1000);
    
    Xtrain=words_train_s(row_sel1,cols_sel);
    Ytrain=genders_train_s(row_sel1);
    Xtest=words_train_s(row_sel2,cols_sel);
    Ytest=genders_train_s(row_sel2);
    
    %templ = templateTree('MaxNumSplits',1);
    %ens = fitensemble(Xtrain,Ytrain,'GentleBoost',200,'Tree');
    %ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree');
    ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree' ); %ens=regularize(ens);
    %ens = fitensemble(Xtrain,Ytrain, 'RobustBoost',300,'Tree','RobustErrorGoal',0.01,'RobustMaxMargin',1);
    %
    [Yhat scores]= predict(ens,Xtest);
    scores
%     Yhat = neural_network(Xtrain, Ytrain, Xtest, Ytest);
%     Yhat = logistic(Xtrain,Ytrain, Xtest,Ytest);
    acc_ens(i)=sum(round(Yhat)==Ytest)/length(Ytest);
    confusionmat(double(Ytest),double(Yhat))
    acc_ens(i)
%     figure;
%     plot(loss(ens,Xtest,Ytest,'mode','cumulative'));
%     xlabel('Number of trees');
%     ylabel('Test classification error');
end
acc_ens
mean(acc_ens)