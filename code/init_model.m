function model = init_model()

load('tree_ensem.mat');
model.tree_ensem = ens;
model.cols_sel = cols_sel;

%load('w_ridge.mat');
%model.w_ridge = w_ridge;

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
