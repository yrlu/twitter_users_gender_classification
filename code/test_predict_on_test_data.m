%% test_predict modified for the NO-TRAINd data case

% Author: Max Lu
% Date: Dec 5

% prepare data:
addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');

 words_train = Xtrain(:, 1:5000);
 words_test = Xtest(:, 1:5000);
 
 words_train_x = [words_train; words_train(1,:); words_train(2,:)];
 words_test_x = words_test;
 test_y = ones(size(words_test_x,1),1);


%% 
 img_feature_train = Xtrain(:,5001:5007);
 img_feature_test = Xtest(:,5001:5007); 
 images_test = Xtest(:,5008:35007);
 

    load('U_mu_vars.mat', 'U', 'mu','vars');
    
    hog_feat = [face_hog nose_hog eyes_hog];
    hog_feat_certain = hog_feat(logical(certain),:);

    % [U mu vars] = pca_1(hog_feat_certain');
    [pca_hog,Xhat,avsq] = pcaApply(hog_feat', U, mu, 1500);
    pca_hog = double(pca_hog');
    
   
    certain_test = certain;

    img_test_x = pca_hog;

load top_data_index.mat cols_sel
% cols_sel=index(1:Nfeatures);
words_train_s = [words_train_x, img_feature_train];
words_test_s = [words_test img_feature_test];
train_x_fs = words_train_s(:, cols_sel);
test_x_fs = words_test_s(:, cols_sel);
% train_x_fs = train_fs;
% test_x_fs = test_fs;
%train_y_fs = Y;

toc

disp('Loading models..');
% load models:


mdl.LogRens= LogRens;
mdl.log_model = log_model;
mdl.logboost_model = logboost_model;
mdl.svm_kernel_n_model = svm_kernel_n_model;
mdl.svm_kernel_model = svm_kernel_model;
mdl.svm_hog_model = svm_hog_model;
mdl.nn =nn;

toc
% make prediction:
disp('Making predictions..');
[~, yhat_log] = a_logistic_predict(mdl.log_model,words_test_x);
[~, yhat_nn] = a_nn_predict(mdl.nn,words_test_x);
[~, yhat_fs] = a_ensemble_trees_predict(mdl.logboost_model, test_x_fs);
toc
[~, yhat_kernel_n] = a_predict_kernelsvm_n(mdl.svm_kernel_n_model, train_x_fs, test_x_fs);
[~, yhat_kernel] = a_predict_kernelsvm(mdl.svm_kernel_model, train_x_fs, test_x_fs);
toc
[yhog, yhat_hog] = a_svm_hog_predict(mdl.svm_hog_model, img_test_x);
% [ylbp, yhat_lbp, svm_lbp_model] = svm_predict(img_lbp_train_x_certain,img_train_y_certain, img_lbp_test_x, test_y);
yhat_hog(logical(~certain_test),:) = 0;
% yhat_lbp(logical(~certain_test),:) = 0;




ypred2 = [yhat_log yhat_fs yhat_nn yhat_hog];
ypred2 = sigmf(ypred2, [2 0]);
yhat_kernel_n = sigmf(yhat_kernel_n, [1.5 0]);
yhat_kernel = sigmf(yhat_kernel, [1.5 0]);
ypred2 = [ypred2 yhat_kernel_n yhat_kernel];


Yhat = predict(test_y, sparse(ypred2), mdl.LogRens, ['-q', 'col']);
disp('Done!');
toc