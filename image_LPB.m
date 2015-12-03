% Author: D.W
% Date: Dec 1

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
%% Detect eyes, nose and mouth.
% % tic
% % face_lbp = zeros(5000,4356);
% % 
% % FaceDetect = vision.CascadeObjectDetector();
% % % NoseDetect = vision.CascadeObjectDetector('Nose');
% % % eyeDetect = vision.CascadeObjectDetector('EyePairBig');
% % for i  = 1:size(train_grey,3)
% % profile = train_grey(:,:, i);
% % bbox        = step(NoseDetect, profile);
% % if ~isempty(bbox)
% %     i
% % % Draw the returned bounding box around the detected face.
% % % videoOut = insertObjectAnnotation(profile,'rectangle',bbox(1,:),'Eyes');
% % % imshow(videoOut);
% % size(bbox)
% % bbox
% % nose = imcrop(profile,bbox(1,:));
% % nose=imresize(nose,[50 50]);
% % [featureVector_nose, hogVisualization_nose] = extractHOGFeatures(nose);
% % nose_hog = [nose_hog;featureVector_nose];
% % else
% %     nose_hog = [nose_hog;zeros(1,900)];
% % end
% % 
% % 
% % bbox        = step(EyeDetect, profile);
% % if ~isempty(bbox)
% % eyes = imcrop(profile,bbox(1,:));
% % eyes=imresize(eyes,[25 100]);
% % [featureVector_eye, hogVisualization_eye] = extractHOGFeatures(nose);
% % eyes_hog = [eyes_hog;featureVector_eye];
% % else
% %     eyes_hog = [eyes_hog; zeros(1, 900)];
% % end
% % 
% % end
% % 
% % train_nose_hog = nose_hog;
% % train_eyes_hog = eyes_hog;
% % 
% % save('train_nose_hog.mat', 'train_nose_hog');
% % save('train_eyes_hog.mat', 'train_eyes_hog');
% % 
% % toc
% % 
%%
tic

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the detector.
% videoFileReader = vision.VideoFileReader('visionface.avi');
% videoFrame      = step(videoFileReader);
certain = ones(size(train_grey,3),1);
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
train_grey_cropped(:,:, i) = profile;

else
    bbox_r            = step(faceDetector, train_r(:,:, i));
    bbox_g            = step(faceDetector, train_g(:,:, i));
    bbox_b            = step(faceDetector, train_b(:,:, i));
    if ~isempty(bbox_r)
        profile = imcrop(profile,bbox_r(1,:));
        profile=imresize(profile,[100 100]);
        train_grey_cropped(:,:, i) = profile;
    elseif ~isempty(bbox_g)
        profile = imcrop(profile,bbox_g(1,:));
        profile=imresize(profile,[100 100]);
        train_grey_cropped(:,:, i) = profile;
    elseif ~isempty(bbox_b)
        profile = imcrop(profile,bbox_b(1,:));
        profile=imresize(profile,[100 100]);
        train_grey_cropped(:,:, i) = profile;
    else
        certain(i) = 0;
    end
end
% imshow(profile), title('Detected face');
end

save('train_grey_cropped.mat', 'train_grey_cropped');
toc
%%
% tic
% % Create a cascade detector object.
% faceDetector = vision.CascadeObjectDetector();
% 
% % Read a video frame and run the detector.
% % videoFileReader = vision.VideoFileReader('visionface.avi');
% % videoFrame      = step(videoFileReader);
% for i  = 1:size(test_grey,3)
% profile = test_grey(:,:, i);
% bbox            = step(faceDetector, profile);
% 
% if ~isempty(bbox)
%     i
% % Draw the returned bounding box around the detected face.
% % videoOut = insertObjectAnnotation(profile,'rectangle',bbox,'Face');
% size(bbox)
% bbox
% profile = imcrop(profile,bbox(1,:));
% profile=imresize(profile,[100 100]);
% test_grey(:,:, i) = profile; 
% else
%     bbox_r            = step(faceDetector, test_r(:,:, i));
%     bbox_g            = step(faceDetector, test_g(:,:, i));
%     bbox_b            = step(faceDetector, test_b(:,:, i));
%     if ~isempty(bbox_r)
%         profile = imcrop(profile,bbox_r(1,:));
%         profile=imresize(profile,[100 100]);
%         test_grey(:,:, i) = profile;
%     elseif ~isempty(bbox_g)
%         profile = imcrop(profile,bbox_g(1,:));
%         profile=imresize(profile,[100 100]);
%         test_grey(:,:, i) = profile;
%     elseif ~isempty(bbox_b)
%         profile = imcrop(profile,bbox_b(1,:));
%         profile=imresize(profile,[100 100]);
%         test_grey(:,:, i) = profile;
%     else
%         certain(5000+i) = 0;
%     end
% end
% % imshow(profile), title('Detected face');
% end
% 
% % save('test_grey_faces.mat', 'test_grey');
% % save('face_certain.mat', 'certain');
% toc

%%
for i = 1:size(train_grey,3)
    i
   imshow(train_grey_cropped(:,:,i)); 
end
% for i = 1:size(test_grey,3)
%     i
%    imshow(test_grey(:,:,i)); 
% end

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



%% Generate LBP features: train_LBP & train_LBP_cr
n = size(train_grey_cropped,3);
train_LBP_cr = zeros(n,59);
%train_hog_vis = [];
for i = 1:size(train_grey_cropped,3)
i
img = train_grey_cropped(:,:,i);
train_LBP_cr(i,:) = extractLBPFeatures(img);
%train_hog = [train_hog;featureVector];
%train_hog_vis = [train_hog_vis;hogVisualization];
% imshow(img); hold on;
% plot(hogVisualization);
end
% save('train_hog.mat', 'train_hog');
% save('train_hog_vis.mat','train_hog_vis');
%% Uncropped LBP 65.24%: train_LBP_cr 63.22%

[accuracy, Ypredicted, Ytest] = cross_validation(train_LBP, train_y, 5, @logistic);
i
accuracy


%% train_hog & train_hog_cr
% Generate HOG features:

train_hog_cr = zeros(5000,4356);
% train_hog_vis = [];
for i = 1:size(train_grey_cropped,3)
i
img = train_grey_cropped(:,:,i);
[featureVector, hogVisualization] = extractHOGFeatures(img);
train_hog_cr(i,:) = featureVector;
%train_hog_vis = [train_hog_vis;hogVisualization];
% imshow(img); hold on;
% plot(hogVisualization);
end
% save('train_hog.mat', 'train_hog');
% save('train_hog_vis.mat','train_hog_vis');

%%
%% use all including undetected: train_hog 66.96%, train_hog_cr  74.88%
[accuracy, Ypredicted, Ytest] = cross_validation(train_hog_cr, train_y, 5, @logistic);
i
accuracy



% %% HOG features classification with logistic regression
% 
% % load('train_hog.mat', 'train_hog');
% load('train_nose_hog.mat', 'train_nose_hog');
% load('train_eyes_hog.mat', 'train_eyes_hog');
% 
% tic
% acc = [];
% % for i = 1:50
% pc = 1:170;
% certain_train = certain(1:5000);
% 
% X =  [train_hog train_nose_hog train_eyes_hog];
% X = X(logical(certain_train),:);
% X = double(X);
% Y = train_y(logical(certain_train),:);
% % Y = train_y;
% addpath('./liblinear');
% 
% % B = TreeBagger(95,X,Y, 'Method', 'classification');
% % RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);
% 
% [accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @logistic);
% i
% accuracy
% acc = [acc;accuracy];
% mean(accuracy)
% % end
% toc

%%
%% train_hog: 70.62%; train_hog_cropped: 80.40%
%  train_LBP: 67%; train_LBP_cr: 64.64%
%  train_hog_cr + train_LBP_cr 80.26%
%  train_hog_cr + train_LBP 80.89% 
% load('face_certain.mat', 'certain');

tic
acc = [];
% for i = 1:50
% pc = 1:170;
certain_train = certain(1:5000);
X = train_hog_cr;
X = X(logical(certain_train),:);
X = double(X);
train_y = [genders_train; genders_train(1); genders_train(2)];
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

%% 200/1000 0.7161 
n = sum(certain_train);
[parts] = make_xval_partition(n, 8);
clc
acc=zeros(8,1);
%acc_nb=zeros(8,1);
%acc_log=zeros(8,1);
%bns = calc_bns(words_train,Y);
X = [train_hog_cr, train_LBP];
X = X(logical(certain_train),:);

new_features = X;
Y = train_y(logical(certain_train),:);
IG=calc_information_gain(Y,new_features, [1:size(new_features,2)],10);

%words_train_s=bsxfun(@times,words_train,bns);
% words_train_s=bsxfun(@times,new_features,IG);
[top_igs, idx]=sort(IG,'descend');
%[top_bans, idx]=sort(bns,'descend'); 84.6% ~ 1000 
for i=1:8
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:2000);
    
    Xtrain=X(row_sel1,cols_sel);
    Ytrain=Y(row_sel1);
    Xtest=X(row_sel2,cols_sel);
    Ytest=Y(row_sel2);
    
    %templ = templateTree('MaxNumSplits',1);
    %ens = fitensemble(Xtrain,Ytrain,'GentleBoost',200,'Tree');
    %ens = fitensemble(Xtrain,Ytrain,'LogitBoost',200,'Tree');
    ens = fitensemble(Xtrain,Ytrain,'AdaBoostM1',35,'Tree' ); %ens=regularize(ens);
    %ens = fitensemble(Xtrain,Ytrain, 'RobustBoost',300,'Tree','RobustErrorGoal',0.01,'RobustMaxMargin',1);
    %
    Yhat= predict(ens,Xtest);
    acc_ens(i)=sum(round(Yhat)==Ytest)/length(Ytest);
    confusionmat(Ytest,Yhat)
    figure;
    plot(loss(ens,Xtest,Ytest,'mode','cumulative'));
    xlabel('Number of trees');
    ylabel('Test classification error');
end
acc_ens
 mean(acc_ens)
