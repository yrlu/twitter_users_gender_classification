%% auto encoder
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN','./DL_toolbox/CNN','./DL_toolbox/SAE');

%% Xcl -normalized Raw data
Y = [genders_train; genders_train(1); genders_train(2)];
Y = Y(logical(certain(1:5000)));
X = grey_imgs(:,:,logical(certain)); 
% [~,~,~,Xed] = convert_to_img(X);
[h w n] = size(X)
X  = reshape(X,[h*w n]);
%X = [X X(:,1) X(:,2)]; % make it 5000 for the batch size 100
%X = X';
n = size(X,1);
X = X(:,1:sum(certain(1:5000)));

Xselected = X(:,1:3000)';
Yselected = Y(1:3000);
%
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

visualize(sae.ae{1}.W{1}(:,2:end)')

%%
model = svmtrain(Ytrain, new_feat, '-t 0 -c 1');
[Yhat acc Yprob] = svmpredict(Ytest, new_feat_test, model);
sum(Yhat==Ytest)/length(Ytest)


%% full code 
dbn = rbm(X);
[new_feat, new_feat_test ] = newFeature_rbm( dbn,train_x,test_x );

%% 
[parts] = make_xval_partition(3000, 5);

acc_ens=zeros(5,1);

acc = zeros(5,1);
for i=1:2
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    %cols_sel=idx(1:7);
    
    Xtrain=Xselected(row_sel1,:);
    Ytrain=Yselected(row_sel1);
    Xtest=Xselected(row_sel2,:);
    Ytest=Yselected(row_sel2);
    
    
    dbn = rbm(Xtrain);
    [new_feat, new_feat_test ] = newFeature_rbm( dbn,Xtrain,Xtest );
    Yhat = logistic( new_feat, Ytrain, new_feat_test, Ytest );
    acc(i) = sum(Yhat==Ytest)/length(Ytest);
    
    %confusionmat(Ytest,Yhat)
end
acc
mean(acc)
% new_feat = new_feat(1:n,:);

%%
new_feat = [new_feat; new_feat(1000:1001,:)];
% X = X(1:n, :);
disp('Neurel network + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(new_feat, Y, 5, @neural_network);
accuracy
mean(accuracy)
toc

%%
addpath('util','NN','CNN','SAE','data');
load data/mnist_uint8;

train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([784 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 0.1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   50;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

output_path='models/SAE_noisy.mat';
save(output_path,'sae');