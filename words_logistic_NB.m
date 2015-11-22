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
word_sel=idx(1:1);
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

%%

train_x = words_train;
train_y = genders_train;
test_x = words_test;
test_y = ones(size(test_x,1), 1);

model = train(train_y, sparse(train_x), ['-s 0', 'col']);
[Yhat] = predict(test_y, sparse(test_x), model, ['-q', 'col']);
% dlmwrite('submit.txt',Yhat, '\n');