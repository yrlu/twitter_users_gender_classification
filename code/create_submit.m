disp('Loading data..');
load('train/genders_train.mat', 'genders_train');
load('train/words_train.mat', 'words_train');
load('test/words_test.mat', 'words_test');

addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');
toc

disp('Preparing data..');


% Separate the data into training set and testing set.
train_x = [words_train; words_train(1,:); words_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2,:)];
test_x = words_test;

% % Features selection 
% Use information gain to select the top features from BOTH word_features
% and image_features.
% The features selection is mainly for ensemble trees use.
Nfeatures = 1000;
disp('Training random forest with selected features..');
words_train_s = [words_train, image_features_train; words_test, image_features_test];
% words_train_s = [words_train, image_features_train];
words_train_s = [words_train_s; words_train_s(1,:); words_train_s(2,:)];
genders_train_s = [genders_train; genders_train(1);genders_train(2)];
IG=calc_information_gain(genders_train,[words_train, image_features_train],[1:size([words_train, image_features_train],2)],10);
[top_igs, index]=sort(IG,'descend');

cols_sel=index(1:Nfeatures);
% prepare data for ensemble trees to train and test.
train_x_fs = words_train_s(:, cols_sel);
train_y_fs = genders_train_s; %? 
temp = [words_test, image_features_test];
test_x_fs =temp(:, cols_sel);

cols_sel_knn = index(1:50);
train_x_knn = words_train_s(:, cols_sel_knn);
train_y_knn = genders_train_s; %?
test_x_knn = temp(:, cols_sel_knn);

% The first thing to do is to train a ensembler, currently we use logistic
% regression. To do that, we seperate the training set into 2 pieces:
% 1) The first piece to train the classifiers 
% 2) The second piece to train the ensembler,e.g. logistic regression.
% There we use $proportion for 1). and the rest for 2).
proportion = 0.8;
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

test_y = ones(size(test_x,1),1);

% **YOUR NEW CLASSIFIER GOES HERE**, please see other acc_{classifier}.m
% and follow the interface. If special data need, please prepare the data
% first as above accordingly.
disp('Building ensemble..');
[~, yhat_log] = acc_logistic_regression(train_x_train, train_y_train, train_x_test, train_y_test);
[~, yhat_nn] = acc_neural_net(train_x_train, train_y_train, train_x_test, train_y_test);
[~, yhat_fs] = acc_ensemble_trees(train_x_fs_train, train_y_fs_train, train_x_fs_test, train_y_test);
[~,yhat_knn] = acc_knn(train_x_knn_train, train_y_knn_train, train_x_knn_test, train_y_test);
%[~,yhat_nb] = predict_MNNB(train_x_train, train_y_train, train_x_test, train_y_test);
% The probabilities produced by the classifiers
ypred = [yhat_log yhat_nn yhat_fs yhat_knn];% yhat_nb];

% Train a log_reg ensembler.
LogRens = train(train_y_test, sparse(ypred), ['-s 0', 'col']);
logRensemble = @(test_x) predict(test_y, sparse(test_x), LogRens, ['-q', 'col']);



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
[~,yhat_knn] = acc_knn(train_x_knn, train_y_knn, test_x_knn, test_y);
%[~,yhat_nb] = predict_MNNB(train_x, train_y, test_x, test_y);

% Use trained ensembler to predict Yhat based on the probabilities
% generated from classifiers.
ypred = [yhat_log yhat_nn yhat_fs yhat_knn]; %yhat_nb];
Yhat = logRensemble(ypred);
YProb = ypred;
Ytest = test_y;