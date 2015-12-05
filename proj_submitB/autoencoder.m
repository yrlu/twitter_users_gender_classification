%% Add path
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN','./DL_toolbox/CNN','./DL_toolbox/SAE');
%%
%% Please run the image_features_extract.m to crop the faces from 
% gray-scale images and get the index of face-detected images. 
% Our 5 fold-cross-validation accuracy is 75.17% 
%% 
%% Set Variables
X = grey_imgs(:,:,logical(certain)); 
[h w n] = size(X)
X  = reshape(X,[h*w n]);
Xtrain = X(:,1:sum(certain(1:5000)))';
Xtest = X(:,1:sum(certain(1:5000)))';

Ytrain = [genders_train; genders_train(1); genders_train(2)]; % we cropped the saved the first two images twice.
Ytrain = Ytrain(logical(certain(1:5000)));

% to make it compatible with 100 batch size
Xtrain = Xtrain(1:3300, :);
Ytrain = Ytrain(1:3300);

%% ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([10000 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.1;
sae.ae{1}.inputZeroMaskedFraction   = 0.8;
sae.ae{1}.scaling_learningRate = 0.95;
sae.nonSparsityPenalty               = 0.1;            %  Non sparsity penalty
sae.dropoutFraction                  = 0.25;            %  Dropout 
opts.numepochs =   50;
opts.batchsize = 100;
sae = saetrain(sae, Xtrain, opts);

% new_feat = nnpredict(sae.ae{1,1}, train_x);
old_feat = [Xtrain ones(size(Xtrain,1),1)];
new_feat = old_feat * sae.ae{1}.W{1}';
old_feat_test = [Xtest ones(size(Xtest,1),1)];
new_feat_test = old_feat_test * sae.ae{1}.W{1}';

% visualize(sae.ae{1}.W{1}(:,2:end)')
Ytest = ones(size(new_feat_test,1),1);
Yhat = logistic( new_feat, Ytrain, new_feat_test, Ytest );
%%


