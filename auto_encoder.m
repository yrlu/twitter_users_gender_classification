%% auto encoder
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
%% Xcl -normalized Raw data
Y = genders_train;
n = size(genders_train,1);
X = [words_train; words_test]; %, image_features_train]; 
sizeX= size(X,1);
Xnorm = X./repmat(range(X)+10e-10,sizeX,1); % - repmat(min(X),sizeX,1)
% Caution, remove uninformative NaN data % for nan - columns
X = Xnorm; %(:,all(~isnan(Xnorm)));   
Y = [Y; Y(1000:1001)];
%%
X = [words_train; words_test]; 
train_x = X(1:n,:);
nullD = train_x(1000:1001,:);
train_x = [train_x; nullD]; % to make # of batches an int. 
test_x  = X(n+1:end,:);

%% 
Xtrain = train_x;
train_x = Xtrain(1:4000,:);
train_y = Y(1:4000,:);
test_x = Xtrain(4001:5000,:);
test_y = Y(4001:5000,:);

%% 
m = size(train_x,2);
addpath('DL_toolbox/util','DL_toolbox/NN','DL_toolbox/CNN','DL_toolbox/SAE');
rand('state',0)
sae = saesetup([m 100]); % length of train_x
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 10;
% sae.ae{1}.weightPenaltyL2 = 1e-2;
sae.ae{1}.scaling_learningRate              =0.8;
% sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   25;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);



%% 
rand('state',0);
nn = nnsetup([size(train_x,2) 100 50 2]);
nn.learningRate = 5;
% nn.momentum    = 0;  
nn.activation_function = 'sigm';
nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
nn.scaling_learningRate = 0.9;
nn.W{1} = sae.ae{1}.W{1};
% nn.dropoutFraction     = 0.1;
% nn.nonSparsityPenalty = 0.001;
opts.numepochs = 120;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn loss] = nntrain(nn, train_x, [train_y, ~train_y], opts);
% new_feat = nnpredict(nn, train_x);
Yhat = nnpredict(nn, test_x);
Yhat = ~(Yhat-1);

err = sum(Yhat ~= test_y)/size(test_y,1);


%% 

[ new_feat, new_feat_test ] = newFeature_rbm( dbn,train_x,test_x );
% new_feat = new_feat(1:n,:);

new_feat = [new_feat; new_feat(1000:1001,:)];
%% X = X(1:n, :);
disp('Neurel network + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(new_feat, Y, 5, @neural_network);
accuracy
mean(accuracy)
toc
