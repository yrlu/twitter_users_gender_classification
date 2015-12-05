
function [Yhat, YProb, model] = acc_kernel_n(train_x, train_y, test_x, test_y)
    
    train_x_fs_train = train_x;
    train_x_fs_test = test_x;
    train_y_fs_train = train_y;
    
    kernel =  @(x,x2) kernel_intersection(x, x2);
    FeatTrain = [train_x_fs_train ];
    FeatTest = [train_x_fs_test];
    FeatTrainNormRows = sqrt(sum(abs(FeatTrain).^2,2));
    FeatTrain = bsxfun(@times, FeatTrain, 1./FeatTrainNormRows);
    FeatTestNormRows = sqrt(sum(abs(FeatTest).^2,2));
    FeatTest = bsxfun(@times, FeatTest, 1./FeatTestNormRows);
    

X = FeatTrain;
Y = train_y_fs_train;
Xtest = FeatTest;
Ytest = test_y;


K = kernel(X, X);
Ktest = kernel(X, Xtest);
model = svmtrain(Y, [(1:size(K,1))' K], '-t 4 -c 1');
[Yhat,~,YProb] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);
    
%     [~, SVMK_info]= kernel_libsvm(FeatTrain,train_y_fs_train,...
%         FeatTest,ones(size(train_x_fs_test,1),1),kernel_intersection2);
%     model=SVMK_info.model;
%     Yhat=SVMK_info.yhat;
%     YProb=SVMK_info.vals;
end
