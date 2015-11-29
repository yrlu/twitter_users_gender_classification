% Normalize the words such that the sum(words,2)=5000 for each row
% Deng, Xiang
%11/28
clear all
close all
load .\data\words_train.mat
load .\data\words_test.mat
words_train_n = words_train ; 
user_mean=mean(words_train_n,2);
words_train_n =bsxfun(@rdivide,words_train_n,user_mean); %normalize, divide each row by the mean of that row

words_test_n = words_test ; 
user_mean=mean(words_test_n,2);
words_test_n =bsxfun(@rdivide,words_test_n,user_mean); %normalize, divide each row by the mean of that row


save('./data/words_test_n.mat', 'words_test_n');
save('./data/words_train_n.mat', 'words_train_n');