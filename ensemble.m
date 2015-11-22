% Author: Max Lu
% Date: Nov 21


function [Yhat] = ensemble(train_x, train_y, test_x, test_y)

addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');

tic

proportion = 0.6;
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



% % Linear Regression
X = train_x;
Y = train_y;
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



predictedY = [NBPredict(train_x_test),KNNPredict(train_x_test),LogRpredict(train_x_test),NNetPredict(train_x_test), RFpredict(train_x_test), LRpredict(train_x_test)];
% predictedY = [LogRpredict(train_x_test),NNetPredict(train_x_test), RFpredict(train_x_test)];

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


% % Linear Regression
X = train_x;
Y = train_y;
Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
LRpredict = @(test_x) sign(sigmf(test_x*Wmap, [2 0])-0.5);
% % ---



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


predictedY_test = [NBPredict(test_x),KNNPredict(test_x),LogRpredict(test_x),NNetPredict(test_x), RFpredict(test_x), LRpredict(test_x)];
% predictedY_test = [LogRpredict(test_x),NNetPredict(test_x), RFpredict(test_x)];

Yhat = str2double(ensembled.predict(predictedY_test));

end