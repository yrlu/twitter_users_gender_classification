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
tic
train_x = [images_train; images_train(1,:); images_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2)];
test_x =  images_test;

[train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);
toc
%%
tic

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
    bbox_r            = step(faceDetector, train_r(:,:, i));
    bbox_g            = step(faceDetector, train_g(:,:, i));
    bbox_b            = step(faceDetector, train_b(:,:, i));
    if ~isempty(bbox_r)
        profile = imcrop(profile,bbox_r(1,:));
        profile=imresize(profile,[100 100]);
        train_grey(:,:, i) = profile;
    elseif ~isempty(bbox_g)
        profile = imcrop(profile,bbox_g(1,:));
        profile=imresize(profile,[100 100]);
        train_grey(:,:, i) = profile;
    elseif ~isempty(bbox_b)
        profile = imcrop(profile,bbox_b(1,:));
        profile=imresize(profile,[100 100]);
        train_grey(:,:, i) = profile;
    else
        certain(i) = 0;
    end
end
% imshow(profile), title('Detected face');
end

% save('train_grey_faces.mat', 'train_grey');
toc
%%
tic
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
    bbox_r            = step(faceDetector, test_r(:,:, i));
    bbox_g            = step(faceDetector, test_g(:,:, i));
    bbox_b            = step(faceDetector, test_b(:,:, i));
    if ~isempty(bbox_r)
        profile = imcrop(profile,bbox_r(1,:));
        profile=imresize(profile,[100 100]);
        test_grey(:,:, i) = profile;
    elseif ~isempty(bbox_g)
        profile = imcrop(profile,bbox_g(1,:));
        profile=imresize(profile,[100 100]);
        test_grey(:,:, i) = profile;
    elseif ~isempty(bbox_b)
        profile = imcrop(profile,bbox_b(1,:));
        profile=imresize(profile,[100 100]);
        test_grey(:,:, i) = profile;
    else
        certain(5000+i) = 0;
    end
end
% imshow(profile), title('Detected face');
end

% save('test_grey_faces.mat', 'test_grey');
% save('face_certain.mat', 'certain');
toc

%%
for i = 1:size(train_grey,3)
    i
   imshow(train_grey(:,:,i)); 
end
for i = 1:size(test_grey,3)
    i
   imshow(test_grey(:,:,i)); 
end

%% Mean men's faces and women's faces:
tic
women_mean = mean(train_grey(:,:,logical(bsxfun(@times, certain(1:5000), train_y==1))),3);
figure;imshow(women_mean);

men_mean = mean(train_grey(:,:,logical(bsxfun(@times, certain(1:5000), train_y==0))),3);
figure;imshow(men_mean);
% faces = cat(3,train_grey, test_grey);
toc
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


%% PCA100 + TreeBagger on cropped faces => 76.21%
tic
acc = [];
% for i = 1:20
pc = 1:100;
certain_train = certain(1:5000);
X = img_scores_faces(1:5000, pc);
X = X(logical(certain_train),:);
Y = train_y(logical(certain_train),:);
% Y = train_y;
addpath('./liblinear');

% B = TreeBagger(95,X,Y, 'Method', 'classification');
% RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @rand_forest);
i
accuracy
acc = [acc;accuracy];
mean(accuracy)
toc
% end


%% PCA170 + Logistic on cropped faces(63%) => 77.19%
% PCA170 + Logistic on cropped faces(rgb 72%) => 75.59%
tic
acc = [];
% for i = 1:50
pc = 1:170;
certain_train = certain(1:5000);
X = img_scores_faces(1:5000, pc);
X = X(logical(certain_train),:);
Y = train_y(logical(certain_train),:);
% Y = train_y;
addpath('./liblinear');

% B = TreeBagger(95,X,Y, 'Method', 'classification');
% RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @logistic);
i
accuracy
acc = [acc;accuracy];
mean(accuracy)
toc
% end


%% RAW 10000pixels + Logistic on cropped faces => 74.20%

tic
certain_train = certain(1:5000);
X = reshape(train_grey, [10000 5000]);
X = X(:, logical(certain_train));
X = X';
Y = train_y(logical(certain_train),:);
addpath('./liblinear');

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @logistic);
accuracy
mean(accuracy)
toc



%% Generate HOG features:

train_hog = [];
train_hog_vis = [];
for i = 1:size(train_grey,3)
i
img = train_grey(:,:,i);
[featureVector, hogVisualization] = extractHOGFeatures(img);
train_hog = [train_hog;featureVector];
train_hog_vis = [train_hog_vis;hogVisualization];
% imshow(img); hold on;
% plot(hogVisualization);
end
% save('train_hog.mat', 'train_hog');
% save('train_hog_vis.mat','train_hog_vis');
%%

test_hog = [];
test_hog_vis = [];
for i = 1:size(test_grey,3)
i
img = test_grey(:,:,i);
[featureVector, hogVisualization] = extractHOGFeatures(img);
test_hog = [test_hog;featureVector];
test_hog_vis = [test_hog_vis;hogVisualization];
% imshow(img); hold on;
% plot(hogVisualization);
end
% save('test_hog.mat', 'test_hog');
% save('test_hog_vis.mat','test_hog_vis');


%% HOG features classification with logistic regression
tic
acc = [];
% for i = 1:50
pc = 1:170;
certain_train = certain(1:5000);
X = train_hog(1:5000, :);
X = X(logical(certain_train),:);
X = double(X);
Y = train_y(logical(certain_train),:);
% Y = train_y;
addpath('./liblinear');

% B = TreeBagger(95,X,Y, 'Method', 'classification');
% RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @logistic);
i
accuracy
acc = [acc;accuracy];
mean(accuracy)
% end
toc




