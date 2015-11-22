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





%%
tic
[n m] = size(words_train);
X = [words_train; words_test];
Y = genders_train;

% PCA features
% X = scores(1:n, 1:4900);
X = X(1:n,:);

% add 2 more samples to make n = 5000;
X = [X;X(1,:);X(1,:)];
Y = [Y;Y(1,:);Y(1,:)];
% X = normc(X);

disp('Ensemble + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @ensemble);
accuracy
mean(accuracy)
toc




%% Generate submit.txt

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



predictedY = [NBPredict(train_x_test),KNNPredict(train_x_test),LRpredict(train_x_test),LogRpredict(train_x_test),NNetPredict(train_x_test), RFpredict(train_x_test)];

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


predictedY_test = [NBPredict(test_x),KNNPredict(test_x),LRpredict(test_x),LogRpredict(test_x),NNetPredict(test_x), RFpredict(test_x)];

Yhat = str2double(ensembled.predict(predictedY_test));
toc
