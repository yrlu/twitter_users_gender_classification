% Author: Max Lu
% Date: Dec 5

tic
%% load models:






% save models:
load('./models/submission/log_ensemble.mat','LogRens');
load('models/submission/log_model.mat', 'log_model');
load('models/submission/logboost_model.mat','logboost_model');
load('models/submission/svm_kernel_n_model.mat', 'svm_kernel_n_model');
load('models/submission/svm_kernel_model.mat', 'svm_kernel_model');
load('models/submission/svm_hog_model.mat', 'svm_hog_model');

mdl.LogRens= LogRens;
mdl.log_model = log_model;
mdl.logboost_model = logboost_model;
mdl.svm_kernel_n_model = svm_kernel_n_model;
mdl.svm_kernel_model = svm_kernel_model;
mdl.svm_hog_model = svm_hog_model;





% prepare data:


tic
disp('Loading data..');
load('train/genders_train.mat', 'genders_train');
addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');
toc

disp('Preparing data..');


% Separate the data into training set and testing set.
Y = [genders_train; genders_train(1); genders_train(2,:)];
train_y = Y;

[words_train_X, words_test_X] = gen_data_words();
words_train_x = words_train_X;
words_test_x = words_test_X;
test_y = ones(size(words_test_x,1),1);
% test_y = Y(idx);

[~,certain,pca_hog] = gen_data_hog();
certain_train = certain(1:5000,:);
certain_test = certain(5001:end,:);
certain_train_x = certain_train;

img_train_y_certain = Y(logical(certain_train), :);

img_train = pca_hog(1:5000,:);
img_train_x_certain = img_train(logical(certain_train), :);
img_train_x = img_train;
img_test_x = pca_hog(5001:end,:);

[~, pca_lbp] = gen_data_lbp();
img_lbp_train = pca_lbp(1:5000,:);
img_lbp_train_x_certain = img_lbp_train(logical(certain_train), :);
img_lbp_train_x = img_lbp_train;
img_lbp_test_x = pca_lbp(5001:end,:);

% % Features selection 
[train_fs, test_fs] = gen_data_words_imgfeat_fs(1000);
train_x_fs = train_fs;
test_x_fs = test_fs;
train_y_fs = Y;

toc

% make prediction:






toc