%% data loading:
%% Load the data first, see data_preprocess.m

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

% Prepare/Load PCA-ed data,  
if exist('eigens','var')~= 1
    if exist('coef.mat','file') ~= 2 
        X = [words_train, image_features_train; words_test, image_features_test]; 
        [coef, scores, eigens] = pca(X);
        save('coef.mat', 'coef');
        save('scores.mat', 'scores');
        save('eigens.mat', 'eigens');
    else 
        load('coef.mat', 'coef');
        load('scores.mat', 'scores');
        load('eigens.mat', 'eigens');
    end
end

%% 
% X = scores(1:n, 1:300);
Y = genders_train;
n = size(genders_train,1);

%% Play with SVM
% Ref: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf

% Transform data to the format of an SVM package
% Vector of reals; 
% Scaling (avoid greater numerics domination)to [-1,1];
% X = [words_train, image_features_train; words_test, image_features_test];
X = scores; 
% Y = genders_train;
% Conduct simple scaling on the data [-1,1]
Xnorm = (X - repmat(min(X),9995,1))./repmat(range(X),9995,1);
% Caution, remove uninformative NaN data % for nan - columns
Xcl = Xnorm(:,all(~isnan(Xnorm)));   

%% Test normailized PCA
% [coefNor, scoresNor, eigensNor] = pca(Xcl);
% Does no better
%

%%
%Consider the RBF kernel with 
% add lib path:
addpath('./libsvm');

X = scores(1:n, 1:320);
Y = genders_train;
X = sparse(X);
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @kernel_libsvm);
accuracy
mean(accuracy)
toc

%Use cross-validation to find the best parameter C and r 
