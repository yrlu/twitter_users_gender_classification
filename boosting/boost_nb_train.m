% Deng, Xiang 11/28/2015
function [ model ] = boost_nb_train( X,Y,opt )
model = train_fastnb(sparse(X), Y, [-1 1]);
end

