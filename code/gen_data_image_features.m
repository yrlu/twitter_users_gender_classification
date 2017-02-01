% Author: Max Lu
% Date: Dec 5

function [train_x, test_x] = gen_data_image_features()
load('train/image_features_train.mat', 'image_features_train');
load('test/image_features_test.mat', 'image_features_test');
train_x = [image_features_train;image_features_train(1,:);image_features_train(2,:)];
test_x = image_features_test;
end
