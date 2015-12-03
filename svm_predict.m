function [Yhat, Yprob] = svm_predict( train_x, train_y, test_x, test_y )
addpath ./libsvm
%  sprintf('-t 2 -c %g',)
disp('training svm...')
model = svmtrain(train_y, train_x, '-t 2 -c 10');
save('./models/svm_hog.mat','model');
[Yhat acc Yprob] = svmpredict(test_y, test_x, model);
if Yhat(1) == 1 && Yprob(1)<0
    Yprob = -Yprob;
elseif Yhat(1) ==0 && Yprob(1)>0
    Yprob = -Yprob;
end
end