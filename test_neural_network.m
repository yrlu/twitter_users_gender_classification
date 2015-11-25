% Author: Max Lu
% Date: Nov 19
% Nov 20 update: plain feature + neural network -> 86.01% on test set.

% add lib path:
addpath('./liblinear');

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

X = [words_train, image_features_train; words_test, image_features_test];
Y = genders_train;

% PCA features
% X = scores(1:n, 1:2000);
X = X(1:n,:);

X = [X;X(1,:);X(1,:)];
Y = [Y;Y(1,:);Y(1,:)];
% X = normc(X);
[n m] = size(words_train);

addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');

% X = X(1:n, :);
disp('Neurel network + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @neural_network);
accuracy
mean(accuracy)
toc





%% 
