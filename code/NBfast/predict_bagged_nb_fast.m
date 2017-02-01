% Deng, Xiang 11/28/2015
function [ Yhat ] = predict_bagged_nb_fast(models, Xtest,M )
Yhats=zeros( size(Xtest,1),M);
for i=1:M    
    Yhats(:,i)=predict_nb_fast(models.NB{i},Xtest);
end
Yhat=round(mode(Yhats,2));
end

