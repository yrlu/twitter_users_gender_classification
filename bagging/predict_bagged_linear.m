% Deng, Xiang 11/28/2015
function [ Yhat,Yhats,Yscore,Yscores ] = predict_bagged_linear(models_linear, Xtest,M )
Yhats=zeros( size(Xtest,1),M);
Yscores=zeros( size(Xtest,1),M);
for i=1:M
    [Yhats(:,i),~, Yscores(:,i)]=liblinear_predict(ones(size(Xtest,1),1),sparse(Xtest),models_linear.linear{i}, [   '-q', 'col']);
    Yhatt=Yhats(:,i);
    Yscoret=Yscores(:,i);
    if Yhatt(1) == 1 && Yscoret(1)<0
        Yscores(:,i) = -Yscores(:,i);
    elseif Yhatt(1) ==0 && Yscoret(1)>0
        Yscores(:,i)  = -Yscores(:,i);
    end
end
Yhat=round(mode(Yhats,2));
Yscore=mean(Yscores,2);
end

