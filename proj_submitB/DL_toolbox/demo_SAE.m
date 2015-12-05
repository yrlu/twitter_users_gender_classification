
addpath('util','NN','CNN','SAE','data');
load data/mnist_uint8;

train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([784 200]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.0;
opts.numepochs =   10;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);

% new_feat = nnpredict(sae.ae{1,1}, train_x);
old_feat = [train_x ones(size(train_x,1),1)];
new_feat = old_feat * sae.ae{1}.W{1}';
old_feat_test = [test_x ones(size(test_x,1),1)];
new_feat_test = old_feat_test * sae.ae{1}.W{1}';

visualize(sae.ae{1}.W{1}(:,2:end)')


output_path='models/SAE.mat';
save(output_path,'sae');