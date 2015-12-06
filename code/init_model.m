function model = init_model()
load('models/submission/log_ensemble.mat','LogRens');
load('models/submission/log_model.mat', 'log_model');
load('models/submission/logboost_model.mat','logboost_model');
load('models/submission/svm_kernel_n_model.mat', 'svm_kernel_n_model');
load('models/submission/svm_kernel_model.mat', 'svm_kernel_model');
load('models/submission/svm_hog_model.mat', 'svm_hog_model');
load('models/submission/nn.mat', 'nn');

model.LogRens = LogRens;
model.log_model = log_model;
model.logboost_model = logboost_model;
model.svm_kernel_n_model = svm_kernel_n_model;
model.svm_kernel_model = svm_kernel_model;
model.svm_hog_model = svm_hog_model;
model.nn = nn;
%load('w_ridge.mat');
%model.w_ridge = w_ridge;

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
