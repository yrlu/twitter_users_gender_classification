function [ acc ] = logistic_critieria( Xtrain,Ytrain,Xtest,Ytest )
acc=1-sum(Ytest~=logistic(Xtrain,Ytrain,Xtest,Yest))/length(yt);
acc>0.8;
end

