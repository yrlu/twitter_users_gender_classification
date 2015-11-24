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
% 
% tic
% disp('Loading data..');
% % Load the data first, see prepare_data.
% load('train/genders_train.mat', 'genders_train');
% load('train/images_train.mat', 'images_train');
% load('train/image_features_train.mat', 'image_features_train');
% load('train/words_train.mat', 'words_train');
% 
% addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
% toc
% 
% 
% disp('Preparing data..');

% 
% proportion = 1;
% 
% X = [words_train; words_train(1,:); words_train(2,:)];
% Y = [genders_train; genders_train(1); genders_train(2,:)];
% train_x = X(train_idx, :);
% train_y = Y(train_idx);
% test_x = X(test_idx, :);
% test_y = Y(test_idx);
% 
% train_x_train=train_x(1:end*proportion,:);
% train_y_train=train_y(1:end*proportion);
% train_x_test = train_x(end*proportion+1:end,:);
% train_y_test = train_y(end*proportion+1:end);
% 
% train_x = train_x_train;
% train_y = train_y_train;



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
opts.numepochs = 10;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples
[nn loss] = nntrain(nn, train_x, [Y, ~Y], opts);
% NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);
toc

[Yhat, YProb] = nnpredict_my(nn, test_x);

end