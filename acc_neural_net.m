% Author: Max Lu
% Date: Nov 23

% This function is compatible with cross_validation.m

% Inputs: 
%   train_x, train_y, test_x, test_y: training and testing data
%   accuracy: the expected accuracy
%   opts: please pass all your options of the classifier here!

% Outputs: 
%   Yhat: The labels predicted, 1 for female, 0 for male, -1 for uncertain,
%       which means the probability of correctly classification is below 
%       "accuracy" for that sample!
%   YProb: This is all the *RAW* outputs of the neural network.

function [Yhat, YProb] = acc_neural_net(train_x, train_y, test_x, test_y, accuracy, opts)
% % neural network
% disp('Training neural network..');
% X=train_x;
% Y=train_y;
% rand('state',0);
% nn = nnsetup([size(X,2) 100 50 2]);
% nn.learningRate = 5;
% nn.activation_function = 'sigm';
% nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
% nn.scaling_learningRate = 0.9;
% opts.numepochs = 3;        %  Number of full sweeps through data
% opts.batchsize = 100;       %  Take a mean gradient step over this many samples
% [nn loss] = nntrain(nn, train_x, [Y, ~Y], opts);
% % NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);
% [Yhat, YProb] = nnpredict_my(nn, test_x);
% return;

printtesterr = 0;

X=train_x;
Y=train_y;
train_x = X;
train_y = [Y, ~Y];
% test_x = testX;
testY = test_y;

rand('state',0);
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
test_err = [];
nn.learningRate = 0.1;

for i = 1:10
[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);

[Yhat_t prob_t] = nnpredict_my(nn, train_x);
if printtesterr==1
[Yhat prob] = nnpredict_my(nn, test_x);
test_err = [test_err; sum(~(Yhat-1) ~= testY)/size(testY,1)];
test_err(end)
end
train_err = [train_err;sum(~(Yhat_t-1) ~= Y)/size(train_y,1)];
i

train_err(end)
end

nn.learningRate = 0.01;

for i = 1:10
[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);

[Yhat_t prob_t] = nnpredict_my(nn, train_x);
train_err = [train_err;sum(~(Yhat_t-1) ~= Y)/size(train_y,1)];
if printtesterr == 1
[Yhat prob] = nnpredict_my(nn, test_x);
test_err = [test_err; sum(~(Yhat-1) ~= testY)/size(testY,1)];
test_err(end)
end
i

train_err(end)
end

nn.learningRate = 0.001;

for i = 1:10
[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);

[Yhat_t prob_t] = nnpredict_my(nn, train_x);
train_err = [train_err;sum(~(Yhat_t-1) ~= Y)/size(train_y,1)];

if printtesterr==1
[Yhat prob] = nnpredict_my(nn, test_x);
test_err = [test_err; sum(~(Yhat-1) ~= testY)/size(testY,1)];
test_err(end)
end 

i

train_err(end)
end

nn.learningRate = 0.0001;

for i = 1:10
[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);
[Yhat_t prob_t] = nnpredict_my(nn, train_x);
train_err = [train_err;sum(~(Yhat_t-1) ~= Y)/size(train_y,1)];
if printtesterr == 1
[Yhat prob] = nnpredict_my(nn, test_x);
test_err = [test_err; sum(~(Yhat-1) ~= testY)/size(testY,1)];
test_err(end)
end
i

train_err(end)
end


save('./models/nn.mat','nn');

[Yhat, YProb] = nnpredict_my(nn, test_x);

end