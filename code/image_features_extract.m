% Author: Max Lu
% Date: Dec 4, 2015
% Modified by Dongni on Dec 4, 2015.

%% Load image data from .mat
% clear
disp('img features...')
tic
% load('train/genders_train.mat', 'genders_train');
% load('train/images_train.mat', 'images_train');
% load('test/images_test.mat', 'images_test');

%% Set variables: train_grey and test_grey are the gray-scale images 
% we use to detect faces on and to do feature extractions. 
% Note: we used the first 2 observations twice to make partition of 
% cross-validation easier. This seems to have little impact on the
% classifier. 
train_x = [images_train; images_train(1,:); images_train(2,:)];
test_x =  images_test;

[train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);
grey_imgs = cat(3, train_grey, test_grey);
% red_imgs = cat(3, train_r, test_r);
% green_imgs = cat(3, train_g, test_r);
% blue_imgs = cat(3, train_b, test_b);

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
    %i
    profile = grey_imgs(:,:,i);
    bbox  = step(faceDetector, profile);
    if ~isempty(bbox) % if any faces detected, get the first one
        profile = imcrop(profile,bbox(1,:));
        profile=imresize(profile,[100 100]);
        grey_imgs(:,:,i) = profile;
%    else 
        img = grey_imgs(:,:, i); % extract HOGs multiple times
    [featureVector, ~] = extractHOGFeatures(img);
    img = imgaussfilt(img);
    img = imresize(img, [50 50]);
    [featureVector2, ~] = extractHOGFeatures(img);
    img = imgaussfilt(img);
    img = imresize(img, [25 25]);
    [featureVector3, ~] = extractHOGFeatures(img);
    face_hog(i,:) = [featureVector featureVector2 featureVector3];
    profile = grey_imgs(:,:, i);
%         bbox_r  = step(faceDetector, red_imgs(:,:, i));
%         bbox_g  = step(faceDetector, green_imgs(:,:, i));
%         bbox_b = step(faceDetector, blue_imgs(:,:, i));
%         if ~isempty(bbox_r)
%             profile = imcrop(profile,bbox_r(1,:));
%             profile=imresize(profile,[100 100]);
%             grey_imgs(:,:, i) = profile;
%         elseif ~isempty(bbox_g)
%             profile = imcrop(profile,bbox_g(1,:));
%             profile=imresize(profile,[100 100]);
%             grey_imgs(:,:, i) = profile;
%         elseif ~isempty(bbox_b)
%             profile = imcrop(profile,bbox_b(1,:));
%             profile=imresize(profile,[100 100]);
%             grey_imgs(:,:, i) = profile;        
        else 
        certain(i) = 0;
      end
    % end
    
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
toc
%%
% save certain_HOG.mat eyes_hog face_hog nose_hog certain
%% Only get the features on face-detected images
 certain_train = certain(1:n_train_grey);
 train_y_certain = train_y(logical(certain_train),:);

% train + test PCA
 Hog_total =  [face_hog nose_hog eyes_hog];
 Hog_total_certain = Hog_total(logical(certain),:);
%
[U, mu, vars] = pca_1(Hog_total_certain');
[YPC, ~, ~] = pcaApply(Hog_total_certain', U, mu, 1500);
YPC = double(YPC');

% Cross Validation 
PC_train = YPC(1:sum(certain(1:5000,:)),:);
%%
% save PCA_u_mu_var.mat U mu vars
% save certain_HOG.mat face_hog nose_hog eyes_hog certain U mu 
[accuracy, Ypredicted, Ytest] = cross_validation(PC_train, train_y_certain, 5, @logistic);



