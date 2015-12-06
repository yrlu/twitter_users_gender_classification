function demo_NN_dropout
addpath('util','NN','CNN','SAE','data');
load data/mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
nn = nnsetup([784 100 10]);
opts.numepochs =  25;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
nn.dropoutFraction = 0; 
[nn, loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
disp(er);




