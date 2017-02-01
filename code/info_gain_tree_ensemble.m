function [yhat, y_scores]  = info_gain_tree_ensemble(trainX, trainY, testX, num_fea)
% info_gain_tree_ensemble fits an ensembled-tree model to data
% yhat = the predicted label of testX
% y_scores 
% trainX, trainY = the data used to train the model
% testX = the unlabled data
% num_fea = number of top-IG features employed for model

% (num_tree = number of trees we want to used)
% 
% Modified by D.W on Nov. 22
% see words_logistic_NB.m
% 

% new_features = [words_train,image_features_train]; %make it X
IG=calc_information_gain(trainY, trainX, [1:size(trainX,2)],10);
[~, idx]=sort(IG,'descend');

%words_train_s= trainX;% words_train;
%words_test_s = [words_test, image_features_test];

cols_sel=idx(1:num_fea);
    
Xtrain=trainX(:,cols_sel);
Ytrain=trainY;
    
Xtest=testX(:,cols_sel);
    
ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree' ); %ens=regularize(ens);
    %ens = fitensemble(Xtrain,Ytrain, 'RobustBoost',300,'Tree','RobustErrorGoal',0.01,'RobustMaxMargin',1);
    %
[yhat, y_scores] = predict(ens,Xtest);