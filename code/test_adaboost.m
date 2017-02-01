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

% Prepare/Load PCA-ed data,  
if exist('eigens','var')~= 1
    if exist('coef.mat','file') ~= 2 
        X = [words_train, image_features_train; words_test, image_features_test]; 
        [coef, scores, eigens] = pca(X);
        save('coef.mat', 'coef');
        save('scores.mat', 'scores');
        save('eigens.mat', 'eigens');
    else 
        load('coef.mat', 'coef');
        load('scores.mat', 'scores');
        load('eigens.mat', 'eigens');
    end
end



%%
[n m] = size(words_train);
X = [words_train, image_features_train; words_test, image_features_test];
Y = genders_train;

% PCA features
% X = scores(1:n, 1:4900);
X = X(1:n,:);

% add 2 more samples to make n = 5000;
X = [X;X(1,:);X(1,:)];
Y = [Y;Y(1,:);Y(1,:)];
% X = normc(X);

disp('Adaboost + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @adaboost);
accuracy
mean(accuracy)
toc


%% using adaboost and cross-validation to find a(t);

[n m] = size(words_train);
% X = [words_train, image_features_train; words_test, image_features_test];
X = [words_train; words_test];
Y = genders_train;

% PCA features
% X = scores(1:n, 1:4900);
X = X(1:n,:);

% add 2 more samples to make n = 5000;
X = [X;X(1,:);X(1,:)];
Y = [Y;Y(1,:);Y(1,:)];
% X = normc(X);

disp('Adaboost + cross-validation');
[accuracy, Ypredicted, Ytest, at] = cross_validation_adaboost(X, Y, 5, @adaboost_predict);
accuracy
mean(accuracy)
at
toc





%% Generate Yhat:
trainx = [words_train, image_features_train];
trainy = genders_train;
testx = [words_test, image_features_test];
testy = ones(size(testx,1), 1);



%% Generate Yhat:

% 

% 
% Yhat = adaboost(trainx, trainy, testx, testy);
% dlmwrite('submit.txt',Yhat,'\n');
tic
[n m] = size(words_train);
X = [words_train, image_features_train; words_test, image_features_test];
Y = genders_train;

% PCA features
% X = scores(1:n, 1:4900);
X = X(1:n,:);

% add 2 more samples to make n = 5000;
X = [X;X(1,:);X(1,:)];
Y = [Y;Y(1,:);Y(1,:)];

p = 0.8;
train_x = X(1:size(X,1)*p,:);
train_y = Y(1:size(X,1)*p);
test_x = X(size(X,1)*p+1:size(X,1),:);
test_y = Y(size(X,1)*p+1:size(X,1));

% test_x = [words_test, image_features_test];
% test_y = zeros(size(test_x, 1),1);





addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');

tic



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
X = X(:, 1:320);
Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
LRpredict = @(test_x) sign(sigmf(test_x(:, 1:320)*Wmap, [2 0])-0.5);
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
opts.numepochs = 300;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn loss] = nntrain(nn, train_x, [Y, ~Y], opts);
NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);


B = TreeBagger(95,train_x,train_y, 'Method', 'classification');
RFpredict = @(test_x) sign(str2double(B.predict(test_x)) - 0.5);


majority = LogRpredict(test_x) ;
yhat = majority > 0;

acc = sum(yhat == test_y)/size(yhat,1);
acc
toc