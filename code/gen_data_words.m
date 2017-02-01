% Author: Max Lu
% Date: Dec 5

function [train_x, test_x] = gen_data_words()
    load('train/words_train.mat', 'words_train');
    load('test/words_test.mat', 'words_test');
    
    train_x = [words_train; words_train(1,:); words_train(2,:)];
    test_x = words_test;
end
