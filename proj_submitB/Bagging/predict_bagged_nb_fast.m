% Deng, Xiang 11/28/2015
function [ Yhat ,Yhats, Yscore, Yscores ] = predict_bagged_nb_fast(models, Xtest,M,bns_selected )
Yhats=zeros( size(Xtest,1),M);
Yscores=zeros( size(Xtest,1),M);
Xtest=bsxfun(@times,Xtest,bns_selected  );
Xtest=round(Xtest);
for i=1:M
    [Yhats(:,i),Yscores(:,i)]=predict_nb_fast(models.NB{i},Xtest);
end
Yhat=round(mode(Yhats,2));
Yscore=mean(Yscores,2);
end

