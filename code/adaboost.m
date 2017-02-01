% Author: Max Lu
% Date: Nov 18


% Assuming we have all the data loaded into memory.
function [Yhat] = adaboost(train_x, train_y, test_x, test_y)
%     ClassTreeEns = fitensemble(train_x,train_y,'LogitBoost',200,'Tree');
%     Yhat = predict(ClassTreeEns,test_x);
addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');

tic



% NavieBayes
% Conduct simple scaling on the data [0,1]
X = train_x;
% sizeX= size(X,1);
% Xnorm = (X - repmat(min(X),sizeX,1))./repmat(range(X),sizeX,1);
% Caution, remove uninformative NaN data % for nan - columns
% Xcl = Xnorm(:,all(~isnan(Xnorm)));   
Xcl = norml(X);

NBModel = fitNaiveBayes(Xcl,train_y);%'Distribution','mvmn');
NBPredict = @(test_x) sign(predict(NBModel,norml(test_x))-0.5);

% KNN
KNNModel = fitcknn(train_x,train_y, 'NumNeighbors',11);
KNNPredict = @(test_x) sign(predict(KNNModel,test_x)-0.5);



% Linear Regression
% X = train_x;
% Y = train_y;
% X = X(:, 1:320);
% Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
% LRpredict = @(test_x) sign(sigmf(test_x(:, 1:320)*Wmap, [2 0])-0.5);
% ---



% Hx{1}=LRpredict;

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


% for i = 1:T
% perm = randperm(n);
% X = X(perm,:);
% Y = Y(perm,:);
% i
% % pc = 1:i*400;
% dropout = 0.05;
% ind = (i-1)*(n*dropout)/T+1:(i-1)*(n*dropout)/T+n*(1-dropout);
% model = train(Y(ind,:), sparse(X(ind,:)), ['-s 0', 'col']);
% LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), model, ['-q', 'col']) - 0.5);
% Hx{i} = LogRpredict;
% end


[n m] = size(train_x);
% T = 2;
% Hx = cell(T, 1);
Hx = {NNetPredict, NBPredict, KNNPredict,LogRpredict};
T = size(Hx, 2);
% Hx = {LRpredict, LogRpredict, LogRpredict2, LogRpredict3, LogRpredict4,LogRpredict5,LogRpredict6,LogRpredict7,LogRpredict8};
Di = ones(size(test_x,1), T)/size(test_x,1);
Z = ones(T,1);
a = ones(T,1);
train_y = sign(train_y-0.5);
test_y = sign(test_y-0.5);
for t = 1:T
    t
    if t~=1
        Di(:,t) = (Di(:,t-1).*exp(-a(t-1)*(test_y.*Hx{t}(test_x))))/Z(t-1);
    end
    et = sum(Di(:,t)'*(test_y~=Hx{t}(test_x)));
    et
    a(t) = 0.5*log((1-et)/(et+10e-9));
    Z(t) = 2*sqrt(et*(1-et));
end

Yhat = zeros(size(test_x,1), 1);
a
for t = 1:T
Yhat = Yhat + a(t)*Hx{t}(test_x);
end

Yhat = Yhat > 0;
toc
end