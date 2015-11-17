%% Read data and save them to corresponding .mat files.

% Training data
genders_train= textread('train/genders_train.txt', '%d');
images_train = csvread('train/images_train.txt');
image_features_train = dlmread('train/image_features_train.txt', ' ');
words_train = dlmread('train/words_train.txt', ' ');

save('train/genders_train.mat', 'genders_train');
save('train/images_train.mat', 'images_train');
save('train/image_features_train.mat', 'image_features_train');
save('train/words_train.mat', 'words_train');



%% Testing data
images_test = csvread('test/images_test.txt');
image_features_test = dlmread('test/image_features_test.txt', ' ');
words_test = dlmread('test/words_test.txt', ' ');

save('test/images_test.mat', 'images_test');
save('test/image_features_test.mat', 'image_features_test');
save('test/words_test.mat', 'words_test');


%% Read packed data to memory
% Training 
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');

%% Testing
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');

