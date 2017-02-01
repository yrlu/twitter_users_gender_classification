% Author: Max Lu
% Date: Nov 18

function [Yhat] = svm(train_x, train_y, test_x, test_y)
%     kg = @(x,x2) kernel_gaussian(x, x2, 2);
%     k3 = @(x,x2) kernel_poly(x, x2, 3);
%     [test_err info] = kernel_libsvm(train_x, train_y*1.0, test_x, test_y, kg);
%     Yhat = info.yhat;

%     model = svmtrain(train_y, train_x, '-c 1 -g 0.07 -b 1');
%     [Yhat, accuracy, prob_values] = svmpredict(test_y, test_x, model, '-b 1'); % run the SVM model on the test data

crange = 10.^[-10:2:4];
% acc=zeros(numel(crange),1);
for i = 1:numel(crange)
    model = svmtrain(train_y, train_x, sprintf('-t 2 -c %g', crange(i)));
    [yhat acc vals] = svmpredict(test_y, test_x, model);
    accur(i) = sum(yhat == test_y)/size(test_y,1);
end
accur
[~, bestc] = max(accur);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

% Train and evaluate SVM classifier using libsvm
model = svmtrain(train_y, train_x, sprintf('-t 2 -c %g', crange(bestc)));
[yhat acc vals] = svmpredict(test_y, test_x, model);

Yhat = yhat;
end