% Author: Max Lu
% Date: Dec 4, 2015
% Modified by Dongni on Dec 4, 2015.

%% Load image data from .mat
% clear
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('test/images_test.mat', 'images_test');

%% Set variables: train_grey and test_grey are the gray-scale images 
% we use to detect faces on and to do feature extractions. 
% Note: we used the first 2 observations twice to make partition of 
% cross-validation easier. This seems to have little impact on the
% classifier. 
train_x = [images_train; images_train(1,:); images_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2)];
test_x =  images_test;

[~, ~, ~, train_grey] = convert_to_img(train_x);
[~, ~, ~, test_grey] = convert_to_img(test_x);
grey_imgs = cat(3, train_grey, test_grey);

n_train_grey = size(train_grey,3);
n_test_grey = size(test_grey,3);
n_total = n_train_grey + n_test_grey;
%% Detect and crop faces, eyes, noses from images, 
% then extract HOG features on them. 
% Preallocate arrays to store extracted HOG features
face_hog = zeros(n_total, 5400);
nose_hog = zeros(n_total, 900);
eyes_hog = zeros(n_total, 792);
% Create cascade detector objects for face, nose and eyes.
faceDetector = vision.CascadeObjectDetector();
NoseDetect = vision.CascadeObjectDetector('Nose');
EyeDetect = vision.CascadeObjectDetector('EyePairSmall');
% Create a binary vector to index the face-detected images 
certain = ones(n_total,1);
% Loop through all gray images
for i  = 1:n_total
    profile = grey_imgs(:,:,i);
    bbox  = step(faceDetector, profile);
    if ~isempty(bbox) % if any faces detected, get the first one
        profile = imcrop(profile,bbox(1,:));
        profile=imresize(profile,[100 100]);
        grey_imgs(:,:,i) = profile;
        img = profile; % extract HOGs multiple times
        [featureVector, ~] = extractHOGFeatures(img);
        img = imgaussfilt(img);
        img = imresize(img, [50 50]);
        [featureVector2, ~] = extractHOGFeatures(img);
        img = imgaussfilt(img);
        img = imresize(img, [25 25]);
        [featureVector3, ~] = extractHOGFeatures(img);
        face_hog(i,:) = [featureVector featureVector2 featureVector3];
    else 
        certain(i) = 0;
    end
    bbox_nose = step(NoseDetect, profile);
    if ~isempty(bbox_nose) % if any nose detected, get the first one
        nose = imcrop(profile,bbox_nose(1,:));
        nose = imresize(nose,[50 50]);
        [nose_hog(i,:),~] = extractHOGFeatures(nose);
    end
    bbox_eye = step(EyeDetect, profile);
    if ~isempty(bbox_eye) % if any pair of eyes detected, get the first one
        eyes = imcrop(profile,bbox_eye(1,:));
        eyes=imresize(eyes,[25 100]);
        [eyes_hog(i,:), ~] = extractHOGFeatures(eyes);
    end
end
%%
save certain_face_imgs grey_imgs certain
%% Only get the features on face-detected images
certain_train = certain(1:n_train_grey);
train_y_certain = train_y(logical(certain_train),:);

%% train + test PCA
Hog_total =  [face_hog nose_hog eyes_hog];
Hog_total_certain = Hog_total(logical(certain),:);
%
[U, mu, vars] = pca_toolbox(Hog_total_certain');
[YPC, ~, ~] = pcaApply(Hog_total_certain', U, mu, 1500);
YPC = double(YPC');

% Cross Validation 
%PC_train = YPC(1:sum(certain(1:5000,:)),:);
%
% save certain_HOG.mat face_hog nose_hog eyes_hog certain U mu 
% [accuracy, Ypredicted, Ytest] = cross_validation(PC_train, train_y_certain, 5, @svm_predict);



