% Data process and save as .mat
% Modified by D.W, Nov 19
% Training data
genders_train= dlmread('../train/genders_train.txt');
images_train = dlmread('../train/images_train.txt');
image_features_train = dlmread('../train/image_features_train.txt');
words_train = dlmread('../train/words_train.txt');

save('../train/genders_train.mat', 'genders_train');
save('../train/images_train.mat', 'images_train');
save('../train/image_features_train.mat', 'image_features_train');
save('../train/words_train.mat', 'words_train');

% Testing data
images_test = dlmread('../test/images_test.txt');
image_features_test = dlmread('../test/image_features_test.txt');
words_test = dlmread('../test/words_test.txt');

save('../test/images_test.mat', 'images_test');
save('../test/image_features_test.mat', 'image_features_test');
save('../test/words_test.mat', 'words_test');

run norm_words.m
