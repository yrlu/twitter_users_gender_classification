% Author: Max Lu
% Date: Dec 2


%%
clear;
tic
disp('Loading data..');
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');

addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');

load('img_coef_faces.mat', 'img_coef_faces');
load('img_scores_faces.mat', 'img_scores_faces');
load('img_eigens_faces.mat', 'img_eigens_faces');
load('face_certain.mat','certain');

load('train_hog_pry.mat', 'train_hog');
load('test_hog_pry.mat', 'test_hog');

load('train_nose_hog.mat', 'train_nose_hog');
load('train_eyes_hog.mat', 'train_eyes_hog');

load('test_nose_hog.mat', 'test_nose_hog');
load('test_eyes_hog.mat', 'test_eyes_hog');
toc






%% estimate the certain predictions:

load('predicted1.mat', 'predicted1');
load('predicted2.mat', 'predicted2');
load('predicted3.mat', 'predicted3');

predicted = [predicted1; predicted2; predicted3];

Yhat = predicted(:,1);
Yprob = predicted(:,2);
Ytest = predicted(:,3);

data = [Yhat Yprob abs(Yprob) Ytest Yhat == Ytest];
correct = data((data(:,5) == 1),:);

thres = 0.8;
sum(data((data(:,3) > thres),5)==1)/sum((data(:,3) > thres))

acc = [];
for i = 1:50
    thres = i/10;
    accuracy = sum(data((data(:,3) > thres),5)==1)/sum((data(:,3) > thres))
    acc = [acc; accuracy];
end
plot(acc)


prop = [];
for i = 1:50
   thres = i/10;
   proportion = size(data((data(:,3) > thres),:),1)/size(data,1);
   prop = [prop; proportion];
end
figure;plot(prop);

% Threshold of 4.0 will yield 25% of testing data of 99% prediction
% accuracy.
%%

