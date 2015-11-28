% Author: Max Lu
% Date: Nov 27



load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');



%% 
train_x = [images_train; images_train(1,:); images_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2)];
test_x =  images_test;

[train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);

%%


% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the detector.
% videoFileReader = vision.VideoFileReader('visionface.avi');
% videoFrame      = step(videoFileReader);
certain = ones(size(train_grey,3)+size(test_grey,3),1);
for i  = 1:size(train_grey,3)
profile = train_grey(:,:, i);
bbox            = step(faceDetector, profile);

if ~isempty(bbox)
    i
% Draw the returned bounding box around the detected face.
% videoOut = insertObjectAnnotation(profile,'rectangle',bbox,'Face');
size(bbox)
bbox
profile = imcrop(profile,bbox(1,:));
profile=imresize(profile,[100 100]);
train_grey(:,:, i) = profile;
else
    certain(i) = 0;
end
% imshow(profile), title('Detected face');
end

% save('train_grey_faces.mat', 'train_grey');

%%
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the detector.
% videoFileReader = vision.VideoFileReader('visionface.avi');
% videoFrame      = step(videoFileReader);
for i  = 1:size(test_grey,3)
profile = test_grey(:,:, i);
bbox            = step(faceDetector, profile);

if ~isempty(bbox)
    i
% Draw the returned bounding box around the detected face.
% videoOut = insertObjectAnnotation(profile,'rectangle',bbox,'Face');
size(bbox)
bbox
profile = imcrop(profile,bbox(1,:));
profile=imresize(profile,[100 100]);
test_grey(:,:, i) = profile; 
else
    certain(5000+i) = 0;
end
% imshow(profile), title('Detected face');
end

% save('test_grey_faces.mat', 'test_grey');
save('face_certain.mat', 'certain');

%% Mean men's faces and women's faces:

women_mean = mean(train_grey(:,:,genders_train==1),3);
figure;imshow(women_mean);

men_mean = mean(train_grey(:,:,genders_train==0),3);
figure;imshow(men_mean);
% faces = cat(3,train_grey, test_grey);

%%


X = cat(3, train_grey, test_grey);
[h w n] = size(X);
x = reshape(X,[h*w n]); 
[img_coef_faces, img_scores_faces, img_eigens_faces] = pca(x');
save('img_coef_faces.mat', 'img_coef_faces');
save('img_scores_faces.mat', 'img_scores_faces');
save('img_eigens_faces.mat', 'img_eigens_faces');


%%
load('img_coef_faces.mat', 'img_coef_faces');
load('img_scores_faces.mat', 'img_scores_faces');
load('img_eigens_faces.mat', 'img_eigens_faces');

%%
[h w n] = size(X);

for i = 1:16
subplot(4,4,i) 
imagesc(reshape(img_coef_faces(:,i),h,w)) 
end

%%

pc = 1:100;
certain_train = certain(1:5000);
X = img_scores_faces(1:5000, pc);
% X = X(logical(certain_train),:);
% Y = train_y(logical(certain_train),:);
Y = train_y;
addpath('./liblinear');

% B = TreeBagger(95,X,Y, 'Method', 'classification');
% RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @rand_forest);
accuracy
mean(accuracy)
toc




