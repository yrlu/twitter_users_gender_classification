%% auto encoder
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
% Xcl -normalized Raw data
Y = genders_train;
n = size(genders_train,1);

train_x = Xcl(1:n,:);
nullD = Xcl(1000:1001,:);
train_x = [train_x; nullD];
test_x  = Xcl(n+1:end,:);
[ dbn ] = rbm( train_x );
[ new_feat, new_feat_test ] = newFeature_rbm( dbn,train_x,test_x );
new_feat = new_feat(1:n,:);

%%
folds = 4;
disp('linear regression + auto-encoder');
X = new_feat;
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, folds, @linear_regression);
accuracy
mean(accuracy)

%%
addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @logistic);
accuracy
mean(accuracy)
toc
%%
addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 4, @kernel_libsvm);
accuracy
mean(accuracy)
toc