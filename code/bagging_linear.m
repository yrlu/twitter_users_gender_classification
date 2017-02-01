 
function [Yhat, YProb, model] = bagging_linear( train_x, train_y, test_x, test_y )

addpath ./bagging

Xtrain_HOG = train_x;
Ytrain_HOG = train_y;

Xtest_HOG = test_x;

   s=0;
   c=0.15;
   F=0.6;% fraction of data for training each model----my rule of thumb: (1-f)^m ~=0.001, too small will overfit
   M_HOG=8; % number of linear models
   
%    size(Xtrain_HOG,2)
   [models_linear_HOG,cols_sel]=train_bag_linear(Xtrain_HOG,Ytrain_HOG,size(Xtrain_HOG,2),0,0,s,c,F,M_HOG);
   % test
   Yhat_HOG_temp=zeros(size(train_y,1))-1;%undefined hog --set predict to -1 so that we will ignore this value in majority voting
   [Yhat_HOG,~,Yscore_HOG,~]=predict_bagged_linear(models_linear_HOG,Xtest_HOG(:, cols_sel),M_HOG);
   Yhat = Yhat_HOG;
   YProb = Yscore_HOG;
   model = models_linear_HOG;
