% D.Wang ; 11/23
%
function [yhat, y_scores] = gaussian_mixture(X,train_y,test_x,test_y, K)
% gaussian_mixture(train_x,train_y,test_x,test_y, K) fits a GMM to train
% data, and predict label for test data based on posterior. 
%
% separate into k clusters and assign labels to each cluster
label = zeros(K,1);

%[IDX,C] = kmeans(train_x, K, 'MaxIter', 100);
gm = fitgmdist(X,K,'RegularizationValue',0.1);
IDX = cluster(gm,X);
for j = 1:K
    c = IDX == j;
    table = tabulate(train_y(c));
    [~,index] = max(table(:,2));
    label(j) = table(index,1)
end

P = posterior(gm,test_x); 
%P is n-by-k, with P(I,J) the probability of component J given observation I.

% assign test points to clusters

% for each point, calculate the prob it belong to a class
prob_class0 = sum(P(:,~logical(label)),2);
prob_class1 = sum(P(:,logical(label)),2);
y_scores = [prob_class0 prob_class1];
%precision = mean(cpre == test_y);
yhat = prob_class1 > prob_class0;