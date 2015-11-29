% Author: Max Lu
% Date: Nov 28


%%

load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');

load('train_grey_faces.mat', 'train_grey');
load('test_grey_faces.mat', 'test_grey');
% load('train_hog.mat', 'train_hog');
% load('test_hog.mat', 'test_hog');

%%


addpath (genpath ('./facerec/m'));


% load data
% [X y width height names] = read_images('/home/philipp/facerec/data/at');

X = train_grey(:,:,500);
y = [genders_train;genders_train(1);genders_train(2)];
y = y(1:100);
width = 100;
height = 100;

% compute a model
fisherface = fisherfaces(X,y');

% plot fisherfaces
figure; hold on;
for i=1:min(16, size(fisherface.W,2))
  subplot(4,4,i);
  comp = cvtGray(fisherface.W(:,i), width, height);
  imshow(comp);
  colormap(jet(256));
  title(sprintf('Fisherface #%i', i));
end


%% 

C = fisherfaces_predict(fisherface, reshape(test_grey, [10000 size(test_grey,3)]), 10);
