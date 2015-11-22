function [ cpre ] = NB( Xtrain, Ytrain, Xtest,Yest )
nb_train = NaiveBayes.fit(Xtrain , Ytrain);
cpre = nb_train.predict(Xtest);
 

end

