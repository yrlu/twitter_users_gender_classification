%% Create a bunch of SVM or LR
% Deng, Xiang

function [ models, cols_sel ] = train_bag_linear(Xtrain,Y,n_ig,n_bns,scale_bns,s,c,F,M )
addpath('./liblinear/proj');
% n_ig: number of top IG used for feature selection
% n_bns: number of top bns used
% scale_bns--scale X by bns score or not
% s: Linear methods option, see blow--can be a group of values
% c: cost parameter
% F: fraction of subset
% M: total number of models

% Feature selection
bns = calc_bns(Xtrain,Y,0.01);
IG=calc_information_gain(Y,Xtrain,[1:size(Xtrain,2)],10);
if (scale_bns)
    bns=bns/max(bns);%normalize bns
    Xtrain =bsxfun(@times,Xtrain,bns);%scale X by bns
end
[top_igs, idx_ig]=sort(IG,'descend');
[top_bns, idx_bns]=sort(bns,'descend');
cols_sel=unique([idx_ig(1:n_ig),idx_bns(1:n_bns)]);
% SVM OR LR options
% 	"Usage: model = train(training_label_vector, training_instance_matrix, 'liblinear_options', 'col');\n"
% 	"liblinear_options:\n"
% 	"-s type : set type of solver (default 1)\n"
% 	"  for multi-class classification\n"
% 	"	 0 -- L2-regularized logistic regression (primal)\n"
% 	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
% 	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
% 	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
% 	"	 4 -- support vector classification by Crammer and Singer\n"
% 	"	 5 -- L1-regularized L2-loss support vector classification\n"
% 	"	 6 -- L1-regularized logistic regression\n"
% 	"	 7 -- L2-regularized logistic regression (dual)\n"
% 	"  for regression\n"
% 	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
% 	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
% 	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
% 	"-c cost : set the parameter C (default 1)\n"
% 	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
% 	"-e epsilon : set tolerance of termination criterion\n"
% 	"	-s 0 and 2\n" 
% 	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
% 	"		where f is the primal function and pos/neg are # of\n" 
% 	"		positive/negative data (default 0.01)\n"
% 	"	-s 11\n"
% 	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
% 	"	-s 1, 3, 4 and 7\n"
% 	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
% 	"	-s 5 and 6\n"
% 	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
% 	"		where f is the primal function (default 0.01)\n"
% 	"	-s 12 and 13\n"
% 	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
% 	"		where f is the dual function (default 0.1)\n"
% 	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
% 	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
% 	"-v n: n-fold cross validation mode\n"
% 	"-q : quiet mode (no outputs)\n"
% 	"col:\n"
% 	"	if 'col' is setted, training_instance_matrix is parsed in column format, otherwise is in row format\n"



Nx=size(Xtrain,1); % Total n

for i=1:M
    option  = sprintf('-s %d -q -c %g', s(randperm(length(s),1)), c);
    ind = randperm(Nx);
    subset_size = floor(Nx * F);
    ind_clip = ind(1:subset_size);
    Y_cur = Y(ind_clip);
    X_cur = Xtrain(ind_clip,:);
    %     % Feature selection
    %     bns = calc_bns(X_cur,Y_cur,0.01);
    %     IG=calc_information_gain(Y_cur,X_cur,[1:size(X_cur,2)],10);
    %     if (scale_bns)
    %         bns=bns/max(top_bns);%normalize bns
    %         X_cur =bsxfun(@times,X_cur,bns);%scale X by bns
    %     end
    %     [top_igs, idx_ig]=sort(IG,'descend');
    %     [top_bns, idx_bns]=sort(bns,'descend');
     % ensemble the top features from both ig and bns
    X_cur=X_cur(:,cols_sel);
    
    
    model_cur   = liblinear_train(Y_cur,sparse( X_cur), option);
    
    models.linear{i}=model_cur; 
    models.cols_sel=cols_sel;
end

end

