
% Author: Max Lu
% Date: Dec 5

tic
clear;
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


% % Features selection 
[train_fs, test_fs] = gen_data_words_imgfeat_fs(1000);
train_x_fs = train_fs;
test_x_fs = test_fs;
train_y_fs = Y;

% The first thing to do is to train a ensembler, currently we use logistic
% regression. To do that, we seperate the training set into 2 pieces:
% 1) The first piece to train the classifiers 
% 2) The second piece to train the ensembler,e.g. logistic regression.
% There we use $proportion for 1). and the rest for 2).
proportion = 0.8;

[train_y_train,train_y_test] = gen_data_separate(train_y, proportion);
[train_x_train,train_x_test] = gen_data_separate(words_train_x, proportion);
[train_x_fs_train,train_x_fs_test] = gen_data_separate(train_x_fs, proportion);

[img_train_x_certain_train,~]  = gen_data_separate(img_train_x_certain, proportion);
[~,img_train_x_test] = gen_data_separate(img_train_x, proportion);
[~,certain_train_test] = gen_data_separate(certain_train_x, proportion);
[img_train_y_train,~] = gen_data_separate(img_train_y_certain, proportion);


toc




% **YOUR NEW CLASSIFIER GOES HERE**, please see other acc_{classifier}.m
% and follow the interface. If special data needed, please prepare the data
% first as above accordingly.
disp('Building ensemble..');
[~, yhat_log] = acc_logistic_regression(train_x_train, train_y_train, train_x_test, train_y_test);
[~, yhat_nn] = acc_neural_net(train_x_train, train_y_train, train_x_test, train_y_test);
[~, yhat_fs] = acc_ensemble_trees(train_x_fs_train, train_y_train, train_x_fs_test, train_y_test);
[~, yhat_kernel_n] = acc_kernel_n(train_x_fs_train, train_y_train, train_x_fs_test, train_y_test);
[~, yhat_kernel] = acc_kernel(train_x_fs_train, train_y_train, train_x_fs_test, train_y_test);
[~, yhat_hog] =svm_predict(img_train_x_certain_train,img_train_y_train, img_train_x_test, train_y_test);

yhat_hog(logical(~certain_train_test),:) = 0;
ypred = [yhat_log yhat_fs yhat_nn yhat_hog];
% ypred = [yhat_log yhat_fs yhat_hog];
ypred = sigmf(ypred, [2 0]);
yhat_kernel_n_1 = sigmf(yhat_kernel_n, [2 0]);
yhat_kernel_1 = sigmf(yhat_kernel, [2 0]);
ypred = [ypred yhat_kernel_n_1 yhat_kernel_1];

% Train a log_reg ensembler.
LogRens = train(train_y_test, sparse(ypred), ['-s 0', 'col']);
save('./models/submission/log_ensemble.mat','LogRens');
logRensemble = @(test_x) predict(test_y, sparse(test_x), LogRens, ['-q', 'col']);

toc

% Here, we re-train the classifiers using the whole training set (in order 
% to achieve better performance). And predict the probabilities on testing
% set

% **YOUR NEW CLASSIFIER GOES HERE**, please see other acc_{classifier}.m
% and follow the interface. If special data need, please prepare the data
% first as above accordingly.
disp('Generating real model and predicting Yhat..');
[~, yhat_log, log_model] = acc_logistic_regression(words_train_x, train_y, words_test_x, test_y);
[~, yhat_nn, nn] = acc_neural_net(words_train_x,train_y,words_test_x,test_y);
[~, yhat_fs, logboost_model] = acc_ensemble_trees(train_x_fs, train_y_fs, test_x_fs, test_y);
[~, yhat_kernel_n, svm_kernel_n_model] = acc_kernel_n(train_x_fs, train_y_fs, test_x_fs, test_y);
[~, yhat_kernel, svm_kernel_model] = acc_kernel(train_x_fs, train_y_fs, test_x_fs, test_y);
% [yhog, yhat_hog, svm_hog_model] = svm_predict(img_train_x_certain,img_train_y_certain, img_test_x, test_y);

disp('training svm...')
addpath('./libsvm')
svm_hog_model = svmtrain(img_train_y_certain, img_train_x_certain, '-t 2 -c 10');
% save('./models/svm_hog.mat','model');
[yhog,~,yhat_hog] = svmpredict(test_y, img_test_x, svm_hog_model);


% [ylbp, yhat_lbp, svm_lbp_model] = svm_predict(img_lbp_train_x_certain,img_train_y_certain, img_lbp_test_x, test_y);
yhat_hog(logical(~certain_test),:) = 0;

% % save models:
save('models/submission/log_model.mat', 'log_model');
save('models/submission/logboost_model.mat','logboost_model');
save('models/submission/svm_kernel_n_model.mat', 'svm_kernel_n_model');
save('models/submission/svm_kernel_model.mat', 'svm_kernel_model');
save('models/submission/svm_hog_model.mat', 'svm_hog_model');
save('models/submission/nn.mat', 'nn');



% generated from classifiers.
ypred2 = [yhat_log yhat_fs yhat_nn yhat_hog];
% ypred2 = [yhat_log yhat_fs yhat_hog];
ypred2 = sigmf(ypred2, [2 0]);
yhat_kernel_n_1 = sigmf(yhat_kernel_n, [1.5 0]);
yhat_kernel_1 = sigmf(yhat_kernel, [1.5 0]);
ypred2 = [ypred2 yhat_kernel_n_1 yhat_kernel_1];


Yhat_log = logRensemble(ypred2);
Yhat = Yhat_log;
YProb = ypred2;
Ytest = test_y;

toc
