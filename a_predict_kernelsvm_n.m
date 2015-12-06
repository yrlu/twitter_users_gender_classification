function [ Yhat, Yscore ] = a_predict_kernelsvm_n( model, Xtrain, Xtest)
kernel =  @(x,x2) kernel_intersection(x, x2);


train_x_fs_train = Xtrain;
train_x_fs_test = Xtest;

FeatTrain = [train_x_fs_train ];
    FeatTest = [train_x_fs_test];
    FeatTrainNormRows = sqrt(sum(abs(FeatTrain).^2,2));
    FeatTrain = bsxfun(@times, FeatTrain, 1./FeatTrainNormRows);
    FeatTestNormRows = sqrt(sum(abs(FeatTest).^2,2));
    FeatTest = bsxfun(@times, FeatTest, 1./FeatTestNormRows);
    

% X = FeatTrain;
% Y = train_y_fs_train;
Xtest = FeatTest;
% Ytest = test_y;

Ktest = kernel(FeatTrain, Xtest);
[Yhat, ~ ,Yscore] = svmpredict(ones(size(Ktest,1),1), [(1:size(Ktest,1))' Ktest], model);
end