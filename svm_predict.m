function [Yhat, Yprob] = svm_predict( train_x, train_y, test_x, test_y )
%  sprintf('-t 2 -c %g',)
disp('training svm...')
model = svmtrain(train_y, train_x, '-t 2 -c 100');
[Yhat acc Yprob] = svmpredict(test_y, test_x, model);
if Yhat(1) == 1 && Yprob(1)<0
    Yprob = -Yprob;
elseif Yhat(1) ==0 && Yprob(1)>0
    Yprob = -Yprob;
end
end