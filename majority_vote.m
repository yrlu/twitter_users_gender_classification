% Author: Max Lu
% Date: Nov 22

% We have to mannually load data here and do whatever each classifier want.
function [Yhat, test_y] = majority_vote(idx)
tic
disp('Loading data..');
% Load the data first, see prepare_data.
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');
load('scores.mat', 'scores');

addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');
toc

disp('Preparing data..');
X = [words_train; words_train(1,:); words_train(2,:)];
Y = [genders_train; genders_train(1); genders_train(2,:)];
train_x = X(~idx, :);
train_y = Y(~idx);
test_x = X(idx, :);
test_y = Y(idx);

% % Logistic Regression
disp('Training logistic regression..');
LogRmodel = train(train_y, sparse(train_x), ['-s 0', 'col']);
% [predicted_label, accuracy, prob_estimates]
LogRpredict = @(test_x) sign(predict(ones(size(test_x,1),1), sparse(test_x), LogRmodel, ['-q', 'col']) - 0.5);



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
NNetPredict = @(test_x) sign(~(nnpredict(nn, test_x)-1) -0.5);
toc


% % Features selection + random forest
disp('Training random forest with selected features..');
words_train_s = [words_train, image_features_train];
words_train_s = [words_train_s; words_train_s(1,:); words_train_s(2,:)];
genders_train_s = [genders_train; genders_train(1);genders_train(2)];
IG=calc_information_gain(genders_train,[words_train, image_features_train],[1:size([words_train, image_features_train],2)],10);
[top_igs, index]=sort(IG,'descend');

cols_sel=index(1:1000);
train_x_fs = words_train_s(~idx, cols_sel);
train_y_fs = genders_train_s(~idx);
test_x_fs = words_train_s(idx, cols_sel);

ens = fitensemble(train_x_fs,train_y_fs,'LogitBoost',200,'Tree' ); 
FSPredict = @(test_x) sign(predict(ens,test_x)-0.5);
toc

disp('Predicting Yhat..');

% % Adaboost
% Hx = {LogRpredict,NNetPredict,FSPredict};
% T = size(Hx, 2);
% Di = ones(size(train_x,1), T)/size(train_x,1);
% Z = ones(T,1);
% a = ones(T,1);
% train_y = sign(train_y-0.5);
% train_x_cell = {train_x, train_x, train_x_fs};
% test_x_cell = {test_x, test_x, test_x_fs};
% for t = 1:T
%     t
%     yhat = Hx{t}(train_x_cell{t});
%     if t~=1
%         Di(:,t) = (Di(:,t-1).*exp(-a(t-1)*(train_y.*yhat)))/Z(t-1);
%     end
%     et = Di(:,t)'*(train_y~=yhat);
%     et
%     a(t) = 0.5*log((1-et)/(et));
% %     Z(t) = 2*sqrt(et*(1-et));
%     Z(t) = sum(Di(:,t).*exp(-a(t)*(train_y.*yhat)));
% end
% 
% Yhat = zeros(size(test_x,1), 1);
% a(a<0) = 0;
% a
% for t = 1:T
% Yhat = Yhat + a(t)*Hx{t}(test_x_cell{t});
% end
% 
% Yhat = Yhat > 0;
% 


Yhat = [LogRpredict(test_x) NNetPredict(test_x) FSPredict(test_x_fs)];
Yhat = mean(Yhat')'>0;
confusionmat(double(test_y),double(Yhat))
toc
end
