% Author: Max Lu
% Date: Nov 19

function [Yhat] = neural_network(X,Y,testX,testY)

train_x = X;
train_y = [Y, ~Y];
test_x = testX;

rand('state',0);
nn = nnsetup([size(X,2) 100 2]);
% nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
opts.numepochs =  50;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn loss] = nntrain(nn, train_x, train_y, opts);
% new_feat = nnpredict(nn, train_x);
Yhat = nnpredict(nn, test_x);
Yhat = ~(Yhat-1)
end



