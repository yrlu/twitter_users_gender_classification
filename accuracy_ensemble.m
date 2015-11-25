% Author: Max Lu
% Date: Nov 23

% This is an updated version of majority_voting.m, we can incorprate more
% classifiers later. This function is compatible with cross_validation_idx.
% If you create new classifiers, please follow the interface below!


% Inputs: 
%   idx: the indices of the testing set. 
%       Here we assume that we have
%       5000 samples (We manually add 2 samples to the training set, so that the
%       neural network will not run into trouble:)
%   accuracy: the expected accuracy
%   opts: please pass all your options of the classifier here!

% Outputs: 
%   Yhat: The labels predicted, 1 for female, 0 for male, -1 for uncertain.
%       Note that we have -1 because we want to limit the accuracy as we
%       inputed
%   Ytest: The test ground truth labels. 
%   YProb: This is all the *RAW* outputs of the classifiers. n*p matrix, n
%   for number of samples, p for number of classifiers. For a single
%   classifier, p = 1.


function [Yhat, Ytest, YProb] = accuracy_ensemble(idx, accuracy, opts)


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


% Separate the data into training set and testing set.
X = [words_train; words_train(1,:); words_train(2,:)];
Y = [genders_train; genders_train(1); genders_train(2,:)];
train_x = X(~idx, :);
train_y = Y(~idx);
test_x = X(idx, :);
test_y = Y(idx);

% % Features selection 
% Use information gain to select the top features from BOTH word_features
% and image_features.
% The features selection is mainly for ensemble trees use.
Nfeatures = 1000;
disp('Training random forest with selected features..');
words_train_s = [words_train, image_features_train];
words_train_s = [words_train_s; words_train_s(1,:); words_train_s(2,:)];
genders_train_s = [genders_train; genders_train(1);genders_train(2)];
IG=calc_information_gain(genders_train,[words_train, image_features_train],[1:size([words_train, image_features_train],2)],10);
[top_igs, index]=sort(IG,'descend');

cols_sel=index(1:Nfeatures);
% prepare data for ensemble trees to train and test.
train_x_fs = words_train_s(~idx, cols_sel);
train_y_fs = genders_train_s(~idx);
test_x_fs = words_train_s(idx, cols_sel);


cols_sel_knn = index(1:350);
train_x_knn = words_train_s(~idx, cols_sel_knn);
train_y_knn = genders_train_s(~idx); %?
test_x_knn = words_train_s(idx, cols_sel_knn);


% The first thing to do is to train a ensembler, currently we use logistic
% regression. To do that, we seperate the training set into 2 pieces:
% 1) The first piece to train the classifiers 
% 2) The second piece to train the ensembler,e.g. logistic regression.
% There we use $proportion for 1). and the rest for 2).
proportion = 0.75;
train_x_train=train_x(1:end*proportion,:);
train_y_train=train_y(1:end*proportion);
train_x_test = train_x(end*proportion+1:end,:);
train_y_test = train_y(end*proportion+1:end);

% Again, we have to split features selection data into two pieces.
train_x_fs_train = train_x_fs(1:end*proportion,:);
train_y_fs_train = train_y_fs(1:end*proportion);
train_x_fs_test = train_x_fs(end*proportion+1:end, :);

% knn
train_x_knn_train = train_x_knn(1:end*proportion,:);
train_y_knn_train = train_y_knn(1:end*proportion);
train_x_knn_test = train_x_knn(end*proportion+1:end, :);


toc









% **YOUR NEW CLASSIFIER GOES HERE**, please see other acc_{classifier}.m
% and follow the interface. If special data needed, please prepare the data
% first as above accordingly.
disp('Building ensemble..');
[~, yhat_log] = acc_logistic_regression(train_x_train, train_y_train, train_x_test, train_y_test);
[~, yhat_nn] = acc_neural_net(train_x_train, train_y_train, train_x_test, train_y_test);
[~, yhat_fs] = acc_ensemble_trees(train_x_fs_train, train_y_fs_train, train_x_fs_test, train_y_test);
[~, yhat_nb] = predict_MNNB(train_x_knn_train, train_y_knn_train, train_x_knn_test, train_y_test);
% The probabilities produced by the classifiers
ypred = [yhat_log yhat_nn yhat_fs yhat_nb];

% Train a log_reg ensembler.
LogRens = train(train_y_test, sparse(ypred), ['-s 0', 'col']);
logRensemble = @(test_x) predict(test_y, sparse(test_x), LogRens, ['-q', 'col']);


toc

% Here, we re-train the classifiers using the whole training set (in order 
% to achieve better performance). And predict the probabilities on testing
% set

% **YOUR NEW CLASSIFIER GOES HERE**, please see other acc_{classifier}.m
% and follow the interface. If special data need, please prepare the data
% first as above accordingly.
disp('Generating real model and predicting Yhat..');
[~, yhat_log] = acc_logistic_regression(train_x, train_y, test_x, test_y);
[~, yhat_nn] = acc_neural_net(train_x,train_y,test_x,test_y);
[~, yhat_fs] = acc_ensemble_trees(train_x_fs, train_y_fs, test_x_fs, test_y);
[~, yhat_nb] = predict_MNNB(train_x_knn, train_y_knn, test_x_knn, test_y);
% Use trained ensembler to predict Yhat based on the probabilities
% generated from classifiers.
ypred = [yhat_log yhat_nn yhat_fs yhat_nb];


%  Fold 1 data, deprecated
%   Probability  thres1   thres2   thres3   Proportion
%     0.9000    0.2000    0.2100    0.3000    0.9910
%     0.9100    0.4000    0.2700    0.4000    0.9780
%     0.9200    1.0000    0.3200    0.5000    0.9580
%     0.9300    1.7000    0.4600    0.7000    0.9230
%     0.9400    2.4000    0.5000    0.9000    0.8880
%     0.9500    3.0000    0.6300    1.1000    0.8490
%     0.9600    4.6000    0.6700    1.3000    0.8020
%     0.9700    9.3000    0.7400    1.6000    0.7230
%     0.9800   10.0000    0.7500    2.0000    0.6550
%     0.9900   10.0000    0.7500    4.2000    0.4390
%     1.0000   10.0000    0.7500    5.8000    0.3800



% Fold 1-5 data:
%   Probability  thres1   thres2   thres3   Proportion
%     0.9000    0.5000    0.3100    0.3000    0.9786
%     0.9100    0.8000    0.4000    0.4000    0.9636
%     0.9200    1.4000    0.4700    0.5000    0.9406
%     0.9300    2.0000    0.5300    0.7000    0.9006
%     0.9400    2.6000    0.6100    0.8000    0.8718
%     0.9500    3.8000    0.6500    1.0000    0.8224
%     0.9600    4.6000    0.7200    1.3000    0.7508
%     0.9700    6.9000    0.7400    1.8000    0.6524
%     0.9800   10.0000    0.7500    2.8000    0.4900
%     0.9900   10.0000    0.7500    7.2000    0.2552
%     1.0000   10.0000    0.7500   10.0000    0.2446

% thres for NB
%   Probability  thres   proportion
%     0.9000    0.0019    0.6208
%     0.9100    0.0021    0.5660
%     0.9200    0.0024    0.5176
%     0.9300    0.0031    0.3707
%     0.9400    0.0040    0.2231
%     0.9500    0.0050    0.1066
%     0.9600    0.0050    0.1066
%     0.9700    0.0050    0.1066
%     0.9800    0.0050    0.1066
%     0.9900    0.0050    0.1066
%     1.0000    0.0050    0.1066


Yhat = acc_cascading(ypred, [4.6,  0.72,  1.3, 0.0024]);
Yuncertain = Yhat==-1;
Ycertain = Yhat~=-1;
Yhat_log = logRensemble(ypred);
Yhat = bsxfun(@times, Yhat, Ycertain)+bsxfun(@times, Yhat_log, Yuncertain);

YProb = ypred;
Ytest = test_y;

toc
end