% Author: Max Lu
% Date: Dec 5

function [Yhat, Ytest, YProb] = accuracy_ensemble(idx)

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
train_y = Y(~idx);

[words_train_X, ~] = gen_data_words();
words_train_x = words_train_X(~idx, :);
words_test_x = words_train_X(idx, :);
test_y = Y(idx);

[~,certain,pca_hog] = gen_data_hog();
certain_train = certain(1:5000,:);
certain_test = certain_train(idx,:);
certain_train_x = certain_train(~idx, :);

img_train_y_certain = Y(logical(bsxfun(@times, ~idx, certain_train)), :);

img_train = pca_hog(1:5000,:);
img_train_x_certain = img_train( logical(bsxfun(@times, ~idx, certain_train)), :);
img_train_x = img_train(~idx, :);
img_test_x = img_train(idx, :);

[~, pca_lbp] = gen_data_lbp();
img_lbp_train = pca_lbp(1:5000,:);
img_lbp_train_x_certain = img_lbp_train( logical(bsxfun(@times, ~idx, certain_train)), :);
img_lbp_train_x = img_lbp_train(~idx,:);
img_lbp_test_x = img_lbp_train(idx, :);

% % Features selection 
[train_fs, ~] = gen_data_words_imgfeat_fs(1000);
train_x_fs = train_fs(~idx, :);
test_x_fs = train_fs(idx,:);
train_y_fs = Y(~idx);

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

[img_lbp_train_x_train,~] = gen_data_separate(img_lbp_train_x_certain, proportion);
[~,img_lbp_train_x_test] = gen_data_separate(img_lbp_train_x, proportion);


toc




% **YOUR NEW CLASSIFIER GOES HERE**, please see other acc_{classifier}.m
% and follow the interface. If special data needed, please prepare the data
% first as above accordingly.
disp('Building ensemble..');
[~, yhat_log] = acc_logistic_regression(train_x_train, train_y_train, train_x_test, train_y_test);
% [~, yhat_nn] = acc_neural_net(train_x_train, train_y_train, train_x_test, train_y_test);
[~, yhat_fs] = acc_ensemble_trees(train_x_fs_train, train_y_train, train_x_fs_test, train_y_test);
[~, yhat_hog] =svm_predict(img_train_x_certain_train,img_train_y_train, img_train_x_test, train_y_test);
[~, yhat_lbp] =svm_predict(img_lbp_train_x_train,img_train_y_train, img_lbp_train_x_test, train_y_test);

yhat_hog(logical(~certain_train_test),:) = 0;
yhat_lbp(logical(~certain_train_test),:) = 0;
ypred = [yhat_log yhat_fs yhat_hog yhat_lbp];
ypred = sigmf(ypred, [2 0]);

% Train a log_reg ensembler.
LogRens = train(train_y_test, sparse(ypred), ['-s 0', 'col']);

save('./models/log_ensemble.mat','LogRens');
logRensemble = @(test_x) predict(test_y, sparse(test_x), LogRens, ['-q', 'col']);


toc

% Here, we re-train the classifiers using the whole training set (in order 
% to achieve better performance). And predict the probabilities on testing
% set

% **YOUR NEW CLASSIFIER GOES HERE**, please see other acc_{classifier}.m
% and follow the interface. If special data need, please prepare the data
% first as above accordingly.
disp('Generating real model and predicting Yhat..');
[~, yhat_log] = acc_logistic_regression(words_train_x, train_y, words_test_x, test_y);
% [~, yhat_nn] = acc_neural_net(words_train_x,train_y,words_test_x,test_y);
[~, yhat_fs] = acc_ensemble_trees(train_x_fs, train_y_fs, test_x_fs, test_y);
[yhog, yhat_hog] = svm_predict(img_train_x_certain,img_train_y_certain, img_test_x, test_y);
[ylbp, yhat_lbp] = svm_predict(img_lbp_train_x_certain,img_train_y_certain, img_lbp_test_x, test_y);
yhat_hog(logical(~certain_test),:) = 0;
yhat_lbp(logical(~certain_test),:) = 0;

% generated from classifiers.
ypred2 = [yhat_log yhat_fs yhat_hog yhat_lbp];
ypred2 = sigmf(ypred2, [2 0]);

Yhat_log = logRensemble(ypred2);
Yhat = Yhat_log;
YProb = ypred2;
Ytest = test_y;
sum(Yhat == test_y)

toc
end