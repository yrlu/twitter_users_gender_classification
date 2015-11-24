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
% NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);


[Yhat, YProb] = nnpredict_my(nn, test_x);

end