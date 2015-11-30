% Author: Max Lu
% Date: Nov 20


%% load data first ..


% Load the data first, see prepare_data.
if exist('genders_train','var')~= 1
prepare_data;
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');
end


%% cross validation 

X = [images_train; images_train(1,:); images_train(2,:)];
Y = [genders_train; genders_train(1); genders_train(2)];
% test_x = images_test;

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @conv_net);
accuracy
mean(accuracy)

%% preprocess the image features
% add 2 more samples 
train_x = [images_train; images_train(1,:); images_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2)];
test_x = images_test;

[train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);

%%
samples= 1:size(train_grey,3);
trainx = double(train_grey(:,:,samples));
testx = double(test_grey);
train_y_tmp = [train_y(samples), ~train_y(samples)];
trainy = double(train_y_tmp');



addpath('./DL_toolbox/util','./DL_toolbox/CNN');


% test_y = double(test_y');


rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 4, 'kernelsize', 5) %convolution layer
};
cnn = cnnsetup(cnn, trainx, trainy);



opts.alpha = 1;
opts.batchsize = 500;
opts.numepochs = 1;

for i=1:50
cnn = cnntrain(cnn, trainx, trainy, opts);
if mod(i,5) ==0
[er, bad] = cnntest(cnn, trainx, trainy);
er
end 
end

%plot mean squared error
figure; plot(cnn.rL);


