% Author: Max Lu
% Date: Nov 19

function [Yhat] = neural_network(X,Y,testX,testY)

train_x = X;
train_y = [Y, ~Y];
test_x = testX;

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
opts.batchsize = 1000;       %  Take a mean gradient step over this many samples

[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);
[Yhat prob] = nnpredict_my(nn, test_x);
Yhat = ~(Yhat-1);
end



