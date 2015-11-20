% Author: Max Lu
% Date: Nov 18


% Assuming we have all the data loaded into memory.
function [Yhat] = adaboost(train_x, train_y, test_x, test_y)
%     ClassTreeEns = fitensemble(train_x,train_y,'LogitBoost',200,'Tree');
%     Yhat = predict(ClassTreeEns,test_x);


X = train_x;
Y = train_y;
X = X(:, 1:320);
Wmap = inv(X'*X+eye(size(X,2))*1e-4) * (X')* Y;
LRpredict = @(test_x) sign(sigmf(test_x(:, 1:320)*Wmap, [2 0])-0.5);


T= 2
Hx = cell(T, 1);
% Hx{1}=LRpredict;

X = train_x;
for i = 1:4
i
pc = 1:i*400;
model = train(Y, sparse(X(:,pc)), ['-s 0', 'col']);
LogRpredict = @(test_x) sign(predict(ones(size(test_x(:,pc),1),1), sparse(test_x(:,pc)), model, ['-q', 'col']) - 0.5);
Hx{i} = LogRpredict;
end


[n m] = size(train_x);
% Hx = {LRpredict, LogRpredict, LogRpredict2, LogRpredict3, LogRpredict4,LogRpredict5,LogRpredict6,LogRpredict7,LogRpredict8};
Di = ones(n, T)/n;
Z = ones(T,1);
a = ones(T,1);
train_y = sign(train_y-0.5);
for t = 1:T
    t
    if t~=1
        Di(:,t) = (Di(:,t-1).*exp(-a(t-1)*(train_y.*Hx{t}(train_x))))/Z(t-1);
    end
    et = sum(Di(:,t)'*(train_y~=Hx{t}(train_x)));
    a(t) = 0.5*log((1-et)/et);
    Z(t) = 2*sqrt(et*(1-et));
end

Yhat = zeros(size(test_x,1), 1);
a
for t = 1:T
Yhat = Yhat + a(t)*Hx{t}(test_x);
end

Yhat = Yhat > 0;
end