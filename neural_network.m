% Author: Max Lu
% Date: Nov 19

function [Yhat] = neural_network(X,Y,testX,testY)

train_x = X;
train_y = [Y, ~Y];
test_x = testX;

rand('state',0);


load('nn.mat');
[Yhat_t prob_t] = nnpredict_my(nn, train_x);
sum(~(Yhat_t-1) ~= Y)/size(train_y,1);

[Yhat prob] = nnpredict_my(nn, testX);
Yhat = ~(Yhat-1);
return 

nn = nnsetup([size(X,2) 100 50 2]);

% nn.momentum    = 0;  
nn.activation_function = 'sigm';
nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
nn.scaling_learningRate = 1;
% nn.dropoutFraction     = 0.1;
% nn.nonSparsityPenalty = 0.001;
opts.numepochs = 5;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

train_err = [];
% test_err = [];
nn.learningRate = 0.1;

for i = 1:10
[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);
[Yhat prob] = nnpredict_my(nn, test_x);
[Yhat_t prob_t] = nnpredict_my(nn, train_x);
% test_err = [test_err; sum(~(Yhat-1) ~= testY)/size(testY,1)];
train_err = [train_err;sum(~(Yhat_t-1) ~= Y)/size(train_y,1)];
i
% test_err(end)
train_err(end)
end

nn.learningRate = 0.01;

for i = 1:10
[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);
[Yhat prob] = nnpredict_my(nn, test_x);
[Yhat_t prob_t] = nnpredict_my(nn, train_x);
% test_err = [test_err; sum(~(Yhat-1) ~= testY)/size(testY,1)];
train_err = [train_err;sum(~(Yhat_t-1) ~= Y)/size(train_y,1)];
i
% test_err(end)
train_err(end)
end

nn.learningRate = 0.001; % 0.0005 % 0.0001

for i = 1:15
[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);
[Yhat prob] = nnpredict_my(nn, test_x);
[Yhat_t prob_t] = nnpredict_my(nn, train_x);
% test_err = [test_err; sum(~(Yhat-1) ~= testY)/size(testY,1)];
train_err = [train_err;sum(~(Yhat_t-1) ~= Y)/size(train_y,1)];
i
% test_err(end)
train_err(end)
end


Yhat = ~(Yhat-1);
end



