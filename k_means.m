function [precision] = k_means(train_x,train_y,test_x,test_y, K)

% separate into k clusters and assign labels to each cluster
label = zeros(K,1);
%[IDX,C] = kmeans(train_x, K, 'MaxIter', 100);
[IDX,C] = kmeans(train_x, K);
for j = 1:K
    c = find(IDX == j);
    table = tabulate(train_y(c));
    [~,index] = max(table(:,2));
    label(j) = table(index,1);
end

% assign test points to clusters
cpre = zeros(size(test_y,1),1);
for j = 1:size(test_y)
    m = bsxfun(@minus,C,double(test_x(j,:)));
    for k = 1:K
        dis(k) = norm(m(k,:));
    end
    [~,ind] = min(dis,[],2);
    cpre(j) = label(ind);
end
precision = mean(cpre == test_y);