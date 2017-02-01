
function [Yhat, YProb, model] = acc_kernel(train_x, train_y, test_x, test_y)

train_x_fs_train = train_x;
train_y_fs_train = train_y;
train_x_fs_test = test_x;

kernel =  @(x,x2) kernel_intersection(x, x2);

X = train_x_fs_train;
Y = train_y_fs_train;
Xtest = train_x_fs_test;
Ytest = test_y;


K = kernel(X, X);
Ktest = kernel(X, Xtest);
model = svmtrain(Y, [(1:size(K,1))' K], '-t 4 -c 0.01');
[Yhat,~,YProb] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);

%     [~,SVMK_infoC]= kernel_libsvm(train_x_fs_train,train_y_fs_train,...
%         train_x_fs_test,ones(size(train_x_fs_test,1),1),kernel_intersection3);
%     model=SVMK_infoC.model;
%     Yhat=SVMK_infoC.yhat;
%     YProb=SVMK_infoC.vals;%
if sum(bsxfun(@times, Yhat, YProb)) < 0
    YProb = -YProb;
end
end
