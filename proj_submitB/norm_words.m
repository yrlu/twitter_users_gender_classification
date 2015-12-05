% Normalize the words such that the sum(words,2)=5000 for each row
% mainly used for NB_fast
% Deng, Xiang
%11/28
clear all
close all
load .\train\words_train.mat
load .\test\words_test.mat
words_train_n = words_train ; 
user_mean=mean(words_train_n,2);
words_train_n =bsxfun(@rdivide,words_train_n,user_mean); %normalize, divide each row by the mean of that row

words_test_n = words_test ; 
user_mean=mean(words_test_n,2);
words_test_n =bsxfun(@rdivide,words_test_n,user_mean); %normalize, divide each row by the mean of that row


save('./train/words_test_n.mat', 'words_test_n');
save('./test/words_train_n.mat', 'words_train_n');