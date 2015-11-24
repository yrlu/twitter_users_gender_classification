% Author: Max Lu
% Date: Nov 20


%% load data first ..


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




%% analyze for the outputs of 6 classifiers:

% load('yy.mat', 'yy');

% [NB,KNN,LogR,NNet, RF, LinearR];

predY = yy(:,1:end-1);
testY = yy(:, end);

ycov = cov(predY);
HeatMap(ycov);


yycor = []
for i = 1:6
    for j = 1:i-1
%         for q = 1:j-1
            yycor = [yycor predY(:,i)*2+predY(:,j)];
%         end
    end
end

yycor = [predY];


correct= bsxfun(@minus, predY, testY) == 0;

[accuracy, Ypredicted, Ytest] = cross_validation(yycor, testY, 5, @rand_forest);
accuracy
mean(accuracy)

%% 5 folds cross-validation yields nice accuracy, see @majority_voting;


tic
% note that here we are calling cross_validation_idx; I leave data
% preparation to each classifier.
disp('Ensemble + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation_idx(5000, 5, @majority_voting);
accuracy
mean(accuracy)
toc



%% Accuracy ensemble, see @accuracy_ensemble;

tic
% note that here we are calling cross_validation_idx; I leave data
% preparation to each classifier.
disp('Accuracy ensemble + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation_idx(5000, 5, @accuracy_ensemble);
accuracy
mean(accuracy)
toc



%% Previous approach, deprecated. Please look the next section.

train_x = [words_train;words_train(1,:);words_train(2,:)];
train_y = [genders_train;genders_train(1,:);genders_train(2,:)];
test_x = words_test;
% test_y = 


addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');

tic

proportion = 0.8;
train_x_train = train_x(1:end*proportion,:);
train_y_train = train_y(1:end*proportion);

train_x_test = train_x(end*proportion+1:end, :);
train_y_test = train_y(end*proportion+1:end, :);

train_x_all = train_x;
train_y_all = train_y;

train_x = train_x_train;
train_y = train_y_train;

% NavieBayes
% Conduct simple scaling on the data [0,1]
X = train_x;
Xcl = norml(X);

NBModel = fitNaiveBayes(Xcl,train_y);%'Distribution','mvmn');
NBPredict = @(test_x) sign(predict(NBModel,norml(test_x))-0.5);

KNNModel = fitcknn(train_x,train_y, 'NumNeighbors',11);
KNNPredict = @(test_x) sign(predict(KNNModel,test_x)-0.5);



% Linear Regression
X = train_x;
Y = train_y;
% X = X(:, :);
Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
LRpredict = @(test_x) sign(sigmf(test_x*Wmap, [2 0])-0.5);
% ---



% % logistc regression
X = train_x;
Y = train_y;
[n m] = size(X);
model = train(Y(:,:), sparse(X(:,:)), ['-s 0', 'col']);
LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']) - 0.5);

% % neural network
rand('state',0);
nn = nnsetup([size(X,2) 100 50 2]);
nn.learningRate = 5;
% nn.momentum    = 0;  
nn.activation_function = 'sigm';
nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
nn.scaling_learningRate = 0.9;
% nn.dropoutFraction     = 0.1;
% nn.nonSparsityPenalty = 0.001;
opts.numepochs = 100;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn loss] = nntrain(nn, train_x, [Y, ~Y], opts);
NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);


B = TreeBagger(95,train_x,train_y, 'Method', 'classification');
RFpredict = @(test_x) sign(str2double(B.predict(test_x)) - 0.5);



predictedY = [NBPredict(train_x_test),KNNPredict(train_x_test),LogRpredict(train_x_test),NNetPredict(train_x_test), RFpredict(train_x_test)];

ensembled = TreeBagger(95,predictedY,train_y_test, 'Method', 'classification');






% train again


train_x = train_x_all;
train_y = train_y_all;

% NavieBayes
% Conduct simple scaling on the data [0,1]
X = train_x;
Xcl = norml(X);

NBModel = fitNaiveBayes(Xcl,train_y);%'Distribution','mvmn');
NBPredict = @(test_x) sign(predict(NBModel,norml(test_x))-0.5);

KNNModel = fitcknn(train_x,train_y, 'NumNeighbors',11);
KNNPredict = @(test_x) sign(predict(KNNModel,test_x)-0.5);



% Linear Regression
X = train_x;
Y = train_y;
% X = X(:, 1:320);
Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
LRpredict = @(test_x) sign(sigmf(test_x*Wmap, [2 0])-0.5);
% ---



% % logistc regression
X = train_x;
Y = train_y;
[n m] = size(X);
model = train(Y(:,:), sparse(X(:,:)), ['-s 0', 'col']);
LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']) - 0.5);

% % neural network
rand('state',0);
nn = nnsetup([size(X,2) 100 50 2]);
nn.learningRate = 5;
% nn.momentum    = 0;  
nn.activation_function = 'sigm';
nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
nn.scaling_learningRate = 0.9;
% nn.dropoutFraction     = 0.1;
% nn.nonSparsityPenalty = 0.001;
opts.numepochs = 100;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn loss] = nntrain(nn, train_x, [Y, ~Y], opts);
NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);


B = TreeBagger(95,train_x,train_y, 'Method', 'classification');
RFpredict = @(test_x) sign(str2double(B.predict(test_x)) - 0.5);

    
predictedY_test = [NBPredict(test_x),KNNPredict(test_x),LogRpredict(test_x),NNetPredict(test_x), RFpredict(test_x)];

Yhat = str2double(ensembled.predict(predictedY_test));
toc



%% generate submit.txt, new benchmark. 90.11%




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


