% Deng, Xiang 11/30/2015
function [ Yhat,Yhats, Yscore,Yscores ] = predict_bagged_tree(models_tree, Xtest,M )
Yhats=zeros( size(Xtest,1),M);
Yscores=zeros( size(Xtest,1),M);
for i=1:M
    [Yhats(:,i), Yscore_tr]=predict(models_tree.enstree{i},Xtest);
    Yscores(:,i)=Yscore_tr(:,2);
end
Yhat=round(mode(Yhats,2));
Yscore=mean(Yscores,2);
end

