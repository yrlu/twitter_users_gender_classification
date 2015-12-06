% Author: Max Lu
% Date: Dec 5

function [train_x_fs, test_x_fs] = gen_data_words_imgfeat_fs(Nfeatures)
load('train/genders_train.mat', 'genders_train');
disp('Selecting words and image_feats features..');
Y = [genders_train;genders_train(1);genders_train(2)];
[words_train_X, words_test_X] = gen_data_words();
[image_features_train, image_features_test] = gen_data_image_features();
% Nfeatures = 1000;
words_train_s = [words_train_X, image_features_train];
words_test_s = [words_test_X, image_features_test];
IG=calc_information_gain(Y,words_train_s,1:size(words_train_s,2),10);
[top_igs, index]=sort(IG,'descend');

cols_sel=index(1:Nfeatures);
train_x_fs = words_train_s(:, cols_sel);
test_x_fs = words_test_s(:, cols_sel);
end