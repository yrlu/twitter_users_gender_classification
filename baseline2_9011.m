% Author: Max Lu
% Date: Nov 23

% Ensemble Logistic Regression, Neural Network, and Ensembled Trees with
% Features Selection. => 90.11%
%% generate submit.txt



tic
disp('Loading data..');
% Load the data first, see prepare_data.
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');
load('scores.mat', 'scores');

addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');
toc

disp('Preparing data..');


proportion = 0.8;

X = [words_train; words_train(1,:); words_train(2,:)];
Y = [genders_train; genders_train(1); genders_train(2,:)];
% train_x = X(~idx, :);
% train_y = Y(~idx);
train_x = X;
train_y = Y;
% test_x = X(idx, :);
% test_y = Y(idx);



train_x_train=train_x(1:end*proportion,:);
train_y_train=train_y(1:end*proportion);
train_x_test = train_x(end*proportion+1:end,:);
train_y_test = train_y(end*proportion+1:end);

train_x = train_x_train;
train_y = train_y_train;

% % Logistic Regression
disp('Training logistic regression..');
LogRmodel = train(train_y, sparse(train_x), ['-s 0', 'col']);
% [predicted_label, accuracy, prob_estimates]
LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), LogRmodel, ['-q', 'col']) - 0.5);



% % neural network
disp('Training neural network..');
X=train_x;
Y=train_y;
rand('state',0);
nn = nnsetup([size(X,2) 100 50 2]);
nn.learningRate = 5;
nn.activation_function = 'sigm';
nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
nn.scaling_learningRate = 0.9;
opts.numepochs = 100;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples
[nn loss] = nntrain(nn, train_x, [Y, ~Y], opts);
NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);
toc


% % Features selection + random forest
disp('Training random forest with selected features..');
words_train_s = [words_train, image_features_train];
words_train_s = [words_train_s; words_train_s(1,:); words_train_s(2,:)];
genders_train_s = [genders_train; genders_train(1);genders_train(2)];
IG=calc_information_gain(genders_train,[words_train, image_features_train],[1:size([words_train, image_features_train],2)],10);
[top_igs, index]=sort(IG,'descend');

cols_sel=index(1:1000);
% train_x_fs = words_train_s(~idx, cols_sel);
% train_y_fs = genders_train_s(~idx);
train_x_fs = words_train_s(:, cols_sel);
train_y_fs = genders_train_s;
% test_x_fs = words_train_s(idx, cols_sel);

train_x_fs_train = train_x_fs(1:end*proportion,:);
train_y_fs_train = train_y_fs(1:end*proportion);

train_x_fs_test = train_x_fs(end*proportion+1:end, :);

train_x_fs = train_x_fs_train;
train_y_fs = train_y_fs_train;


ens = fitensemble(train_x_fs,train_y_fs,'LogitBoost',200,'Tree' ); 
FSPredict = @(test_x) sign(predict(ens,test_x)-0.5);
toc

disp('Building ensemble..');



[predicted_label, accuracy, yhat_logr] = predict(train_y_test, sparse(train_x_test), LogRmodel, ['-q', 'col']);
[Yhat yhat_nn] = nnpredict_my(nn, train_x_test);
yhat_nn = yhat_nn(:,1)-yhat_nn(:,2);
[Yhat yhat_fs]= predict(ens,train_x_fs_test);

ypred = [yhat_logr yhat_nn yhat_fs];

LogRens = train(train_y_test, sparse(ypred), ['-s 0', 'col']);
% [predicted_label, accuracy, prob_estimates]
logRensemble = @(test_x) predict(ones(size(test_x,1),1), sparse(test_x), LogRens, ['-q', 'col']);

toc








disp('Generating real model..');




X = [words_train; words_train(1,:); words_train(2,:)];
Y = [genders_train; genders_train(1); genders_train(2,:)];
train_x = X;
train_y = Y;
% train_x = X(~idx, :);
% train_y = Y(~idx);
% test_x = X(idx, :);
% test_y = Y(idx);


% 
% train_x_train=train_x(1:end*proportion,:);
% train_y_train=train_y(1:end*proportion);
% train_x_test = train_x(end*proportion+1:end,:);
% train_y_test = train_y(end*proportion+1:end);
% 
% train_x = train_x_train;
% train_y = train_y_train;

% % Logistic Regression
disp('Training logistic regression..');
LogRmodel = train(train_y, sparse(train_x), ['-s 0', 'col']);
% [predicted_label, accuracy, prob_estimates]
LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), LogRmodel, ['-q', 'col']) - 0.5);



% % neural network
disp('Training neural network..');
X=train_x;
Y=train_y;
rand('state',0);
nn = nnsetup([size(X,2) 100 50 2]);
nn.learningRate = 5;
nn.activation_function = 'sigm';
nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
nn.scaling_learningRate = 0.9;
opts.numepochs = 100;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples
[nn loss] = nntrain(nn, train_x, [Y, ~Y], opts);
NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);
toc


% % Features selection + random forest
disp('Training random forest with selected features..');
words_train_s = [words_train, image_features_train];
words_train_s = [words_train_s; words_train_s(1,:); words_train_s(2,:)];
genders_train_s = [genders_train; genders_train(1);genders_train(2)];
IG=calc_information_gain(genders_train,[words_train, image_features_train],[1:size([words_train, image_features_train],2)],10);
[top_igs, index]=sort(IG,'descend');

cols_sel=index(1:1000);
% train_x_fs = words_train_s(~idx, cols_sel);
% train_y_fs = genders_train_s(~idx);
train_x_fs = words_train_s(:, cols_sel);
train_y_fs = genders_train_s;
test_x_fs = words_train_s(:, cols_sel);



% train_x_fs_train = train_x_fs(1:end*proportion,:);
% train_y_fs_train = train_y_fs(1:end*proportion);
% 
% train_x_fs_test = train_x_fs(end*proportion+1, :);
% 
% train_x_fs = train_x_fs_train;
% train_y_fs = train_y_fs_train;

ens = fitensemble(train_x_fs,train_y_fs,'LogitBoost',200,'Tree' ); 
FSPredict = @(test_x) sign(predict(ens,test_x)-0.5);
toc



disp('Predicting Yhat..');


test_x = words_test;
test_x_fs = [words_test, image_features_test];
test_x_fs = test_x_fs(:, cols_sel);

[predicted_label, accuracy, yhat_logr] = predict(ones(size(test_x,1),1), sparse(test_x), LogRmodel, ['-q', 'col']);
[Yhat yhat_nn] = nnpredict_my(nn, test_x);
yhat_nn = yhat_nn(:,1)-yhat_nn(:,2);
[Yhat yhat_fs]= predict(ens,test_x_fs);

ypred = [yhat_logr yhat_nn yhat_fs];

% LogRens = train(train_y_test, ypred, ['-s 0', 'col']);
% [predicted_label, accuracy, prob_estimates]
% ensemble = @(test_x) predict(ones(size(test_x,1),1), sparse(test_x), LogRens, ['-q', 'col']);

Yhat = logRensemble(ypred);
% dlmwrite('submit.txt', Yhat, '\n');
