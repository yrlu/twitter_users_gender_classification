clear all
close all
tic
disp('Loading data..');
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');

addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');
toc

disp('Preparing data..');

[n, ~] = size(words_train);


%bns = calc_bns(words_train,Y);
IG=calc_information_gain(genders_train,words_train,[1:5000],10);
[top_bans, idx]=sort(IG,'descend');
%words_train_s=bsxfun(@times,words_train,IG);
%% 
acc = zeros(8,2);
[parts] = make_xval_partition(n, 8);
for i=1:8
    fprintf('Testing fold # %d \n', i); 
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:500);
    
    Xtrain=words_train(row_sel1,cols_sel);
    Ytrain=genders_train(row_sel1);
    Xtest=words_train(row_sel2,cols_sel);
    Ytest=genders_train(row_sel2);
    
%     Xtrain=image_features_train(row_sel1,:);
%     Ytrain=genders_train(row_sel1);
%     Xtest=image_features_train(row_sel2,:);
%     Ytest=genders_train(row_sel2);
    
    % test 20-200 clusters
    for j = 1:40
        fprintf('Testing cluster # %d \n', j); 
        [Yhat,~] = gaussian_mixture(Xtrain,Ytrain,Xtest,Ytest, j*5);
        confusionmat(Ytest,Yhat)
        acc(i,j)=sum(round(Yhat)==Ytest)/length(Ytest);
    end
end
acc;
mean(acc);