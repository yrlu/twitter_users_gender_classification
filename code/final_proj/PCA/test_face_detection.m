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


%% Detect eyes, nose and mouth.
tic
n = size(train_Grey,3);
nose_hog = zeros(n,900);
eyes_hog = zeros(n,900);

% MouthDetect = vision.CascadeObjectDetector('Mouth');
NoseDetect = vision.CascadeObjectDetector('Nose');
EyeDetect = vision.CascadeObjectDetector('EyePairBig');
for i  = 1:size(train_grey,3)
profile = train_grey(:,:, i);
bbox        = step(NoseDetect, profile);
if ~isempty(bbox)
    i
% Draw the returned bounding box around the detected face.
% videoOut = insertObjectAnnotation(profile,'rectangle',bbox(1,:),'Eyes');
% imshow(videoOut);
size(bbox)
bbox
nose = imcrop(profile,bbox(1,:));
nose=imresize(nose,[50 50]);
[featureVector_nose, hogVisualization_nose] = extractHOGFeatures(nose);
nose_hog(i,:) = featureVector_nose;
end


bbox        = step(EyeDetect, profile);
if ~isempty(bbox)
eyes = imcrop(profile,bbox(1,:));
eyes=imresize(eyes,[25 100]);
[featureVector_eye, hogVisualization_eye] = extractHOGFeatures(nose);
eyes_hog(i,:) = featureVector_eye;
else
    eyes_hog = [eyes_hog; zeros(1, 900)];
end

end

train_nose_hog = nose_hog;
train_eyes_hog = eyes_hog;

save('train_nose_hog.mat', 'train_nose_hog');
save('train_eyes_hog.mat', 'train_eyes_hog');

toc


%%

tic

nose_hog = [];
eyes_hog = [];

% MouthDetect = vision.CascadeObjectDetector('Mouth');
NoseDetect = vision.CascadeObjectDetector('Nose');
EyeDetect = vision.CascadeObjectDetector('EyePairBig');
for i  = 1:size(test_grey,3)
profile = test_grey(:,:, i);
bbox        = step(NoseDetect, profile);
if ~isempty(bbox)
    i
% Draw the returned bounding box around the detected face.
% videoOut = insertObjectAnnotation(profile,'rectangle',bbox(1,:),'Eyes');
% imshow(videoOut);
%size(bbox)
%bbox
nose = imcrop(profile,bbox(1,:));
nose=imresize(nose,[50 50]);
[featureVector_nose, hogVisualization_nose] = extractHOGFeatures(nose);
nose_hog = [nose_hog;featureVector_nose];
else
    nose_hog = [nose_hog;zeros(1,900)];
end


bbox        = step(EyeDetect, profile);
if ~isempty(bbox)
eyes = imcrop(profile,bbox(1,:));
eyes=imresize(eyes,[25 100]);
[featureVector_eye, hogVisualization_eye] = extractHOGFeatures(nose);
eyes_hog = [eyes_hog;featureVector_eye];
else
    eyes_hog = [eyes_hog; zeros(1, 900)];
end

end

test_nose_hog = nose_hog;
test_eyes_hog = eyes_hog;

save('test_nose_hog.mat', 'test_nose_hog');
save('test_eyes_hog.mat', 'test_eyes_hog');

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

end
% imshow(profile), title('Detected face');
end

save('train_grey_faces.mat', 'train_grey');
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

 save('test_grey_faces.mat', 'test_grey');
 save('face_certain.mat', 'certain');
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




%% Generate LBP features:

% load('train_grey_faces.mat', 'train_grey');
train_lbp = zeros(5000, 59);
for i = 1:size(train_grey, 3)
i
img = train_grey(:,:,i);
lbp_feat = extractLBPFeatures(img);
train_lbp(i,:) = lbp_feat;
end 
save('train_lbp.mat', 'train_lbp');



%%

% load('test_grey_faces.mat', 'test_grey');
test_lbp = zeros(4997, 59);
for i = 1:size(test_grey, 3)
i
img = test_grey(:,:,i);
lbp_feat = extractLBPFeatures(img);
test_lbp(i,:) = lbp_feat;
end 
save('test_lbp.mat', 'test_lbp');


%% Generate HOG features:


train_hog = zeros(5000, 5400);
% train_hog_vis = [];
for i = 1:size(train_grey,3)
i
img = histeq(train_grey(:,:,i));
[featureVector, hogVisualization] = extractHOGFeatures(img);




img = imgaussfilt(img);
img = imresize(img, [50 50]);

[featureVector2, hogVisualization] = extractHOGFeatures(img);

img = imgaussfilt(img);
img = imresize(img, [25 25]);

[featureVector3, hogVisualization] = extractHOGFeatures(img);


img = imgaussfilt(img);
img = imresize(img, [12 12]);

[featureVector4, hogVisualization] = extractHOGFeatures(img);


img = imgaussfilt(img);
img = imresize(img, [8 8]);

[featureVector5, hogVisualization] = extractHOGFeatures(img);



img = imgaussfilt(img);
img = imresize(img, [4 4]);

[featureVector6, hogVisualization] = extractHOGFeatures(img);



train_hog(i,:) = [featureVector featureVector2 featureVector3 featureVector4 featureVector5 featureVector6];
% train_hog_vis = [train_hog_vis;hogVisualization];
% imshow(img); hold on;
% plot(hogVisualization);
end
 save('train_hog.mat', 'train_hog');
% save('train_hog_vis.mat','train_hog_vis');
%%
tic
test_hog = zeros(4997, 5400);
test_hog_vis = [];
for i = 1:size(test_grey,3)
%i
img = histeq(test_grey(:,:,i)); 
[featureVector, hogVisualization] = extractHOGFeatures(img);


img = imgaussfilt(img);
img = imresize(img, [50 50]);

[featureVector2, hogVisualization] = extractHOGFeatures(img);

img = imgaussfilt(img);
img = imresize(img, [25 25]);

[featureVector3, hogVisualization] = extractHOGFeatures(img);


img = imgaussfilt(img);
img = imresize(img, [12 12]);

[featureVector4, hogVisualization] = extractHOGFeatures(img);


img = imgaussfilt(img);
img = imresize(img, [8 8]);

[featureVector5, hogVisualization] = extractHOGFeatures(img);



img = imgaussfilt(img);
img = imresize(img, [4 4]);

[featureVector6, hogVisualization] = extractHOGFeatures(img);

test_hog(i,:) = [featureVector featureVector2 featureVector3 featureVector4 featureVector5 featureVector6];
% test_hog_vis = [test_hog_vis;hogVisualization];
% imshow(img); hold on;
% plot(hogVisualization);
end
toc
 save('test_hog.mat', 'test_hog');
% save('test_hog_vis.mat','test_hog_vis');




%% PCA

load('face_certain.mat', 'certain');
load('train_nose_hog.mat', 'train_nose_hog');
load('train_eyes_hog.mat', 'train_eyes_hog');
load('train_hog.mat', 'train_hog');
load('train/genders_train.mat', 'genders_train');

%%
train_y = [genders_train; genders_train(1); genders_train(2)];
tic
certain_train = certain(1:5000);

X =  [train_hog train_nose_hog train_eyes_hog];
% X = train_hog;
% X = X(logical(certain_train),:);
X = double(X);
Y = train_y(logical(certain_train),:);


%% Train+test PCA
load('face_certain.mat', 'certain');
load('train_nose_hog.mat', 'train_nose_hog');
load('train_eyes_hog.mat', 'train_eyes_hog');
load('train_hog.mat', 'train_hog');
load('./train/genders_train.mat', 'genders_train');
load('test_hog.mat', 'test_hog');
load('test_nose_hog.mat', 'test_nose_hog');
load('test_eyes_hog.mat', 'test_eyes_hog');


%%
addpath('./PCA');
certain_train = certain(1:5000);

traintest =  [train_hog train_nose_hog train_eyes_hog ; test_hog test_nose_hog test_eyes_hog];
traintest_certain = traintest(logical(certain),:);
[U mu vars] = pca_toolbox(traintest_certain');

%%
n = cumsum(vars)/sum(vars);
plot(n);
%%
addpath('./libsvm')
tic
[YPC,Xhat,avsq] = pcaApply(traintest_certain', U, mu, 2000);
YPC = double(YPC');
PC_train = YPC(1:sum(certain(1:5000,:)),:);
train_y = [genders_train; genders_train(1); genders_train(2)];
train_y_certain = train_y(logical(certain(1:5000)),:);
%%
[accuracy, Ypredicted, Ytest] = cross_validation(PC_train, train_y_certain, 5, @svm_predict);
accuracy
mean(accuracy)
toc

%%
certain_train = certain(1:5000);
tic
train_known =  [train_hog train_nose_hog train_eyes_hog];
train_known_certain = train_known(logical(certain_train),:);
genders_certain = train_y(logical(certain_train));
%%
[U_f mu_f vars_f] = pca_toolbox(train_known_certain(logical(genders_certain(1:3000)),:)');
[U_m mu_m vars_m] = pca_toolbox(train_known_certain(~logical(genders_certain(1:3000)),:)');

[YPC,Xhat,avsq] = pcaApply(train_known_certain', U_f, mu_f, 1500);
[YPC_2,~,~]= pcaApply(train_known_certain', U_m, mu_m, 1500);
%%

YPC_total = [YPC', YPC_2'];
addpath('../')

[yhat_after] = logistic( YPC_total(1:3000,:), genders_certain(1:3000), YPC_total(3001:end,:), genders_certain(3001:end) );

acc =sum(yhat_after==genders_certain(3001:end))/size(genders_certain(3001:end),1);
toc


%% PCA

% load('train_hog.mat', 'train_hog');
load('face_certain.mat', 'certain');
load('train_nose_hog.mat', 'train_nose_hog');
load('train_eyes_hog.mat', 'train_eyes_hog');
load('train_hog.mat', 'train_hog');
%load('./train/genders_train.mat', 'genders_train');

load('test_hog.mat', 'test_hog');
load('test_nose_hog.mat', 'test_nose_hog');
load('test_eyes_hog.mat', 'test_eyes_hog');



train_y = [genders_train; genders_train(1); genders_train(2)];
tic
acc = [];
% for i = 1:50
certain_train = certain(1:5000);

X =  [train_hog train_nose_hog train_eyes_hog];
% X = train_hog;
% X = X(logical(certain),:);
X = double(X);
X_certain = X(logical(certain_train),:);
Y = train_y(logical(certain_train),:);
% Y = train_y;
addpath('./liblinear');

%%
% EXAMPLE
%  load pcaData;
%  [U,mu,vars] = pca( I3D1(:,:,1:12) );
%  [Y,Xhat,avsq] = pcaApply( I3D1(:,:,1), U, mu, 5 );
%  pcaVisualize( U, mu, vars, I3D1, 13, [0:12], [], 1 );
%  Xr = pcaRandVec( U, mu, vars, 1, 25, 0, 3 );
testX = [test_hog test_nose_hog test_eyes_hog];
[U mu vars] = pca_toolbox(X_certain');
% [YPC,Xhat,avsq] = pcaApply(testX', U, mu, 2000 );
[YPC,Xhat,avsq] = pcaApply(X_certain', U, mu, 2000 );
YPC = double(YPC');
size(Xhat)
% size(scores)
% scores_train = scores(1:5000,:);
% scores_train_certain = scores_train(logical(certain_train), :);
toc



%%
% load('img_pca_basis.mat', 'U', 'mu', 'vars');
certain_train = certain(1:5000);
X =  [train_hog train_nose_hog train_eyes_hog];
X_certain = X(logical(certain_train),:);
Y = train_y(logical(certain_train),:);

% B = TreeBagger(95,X,Y, 'Method', 'classification');
% RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);
train_y = [genders_train; genders_train(1); genders_train(2)];
Y = train_y(logical(certain_train),:);
% YPC_certain = YPC(logical(certain_train), :);
% size(YPC_certain)
size(Y)
%%
addpath('../libsvm');
%%
[YPC,Xhat,avsq] = pcaApply(X_certain', U, mu, 1500);
YPC = double(YPC');

%%
[accuracy, Ypredicted, Ytest] = cross_validation(YPC, Y, 5, @svm_predict);
% i
accuracy
% acc = [acc;mean(accuracy)];
mean(accuracy)
% end
toc



%% PCA over both training and testing set:

load('face_certain.mat', 'certain');
load('train_nose_hog.mat', 'train_nose_hog');
load('train_eyes_hog.mat', 'train_eyes_hog');
load('train_hog.mat', 'train_hog');
load('train/genders_train.mat', 'genders_train');

load('test_hog.mat', 'test_hog');
load('test_nose_hog.mat', 'test_nose_hog');
load('test_eyes_hog.mat', 'test_eyes_hog');

train_y = [genders_train; genders_train(1); genders_train(2)];
tic
acc = [];
certain_train = certain(1:5000);

%%
X =  [train_hog train_nose_hog train_eyes_hog train_lbp; test_hog test_nose_hog test_eyes_hog test_lbp];
% X = train_hog;
% X = X(logical(certain),:);
X_certain = X(logical(certain),:);
Y = train_y(logical(certain_train),:);
% Y = train_y;
addpath('./liblinear');
[U, mu vars] = pca_toolbox(X_certain');
% [YPC,Xhat,avsq] = pcaApply(testX', U, mu, 2000 );
[scores,Xhat,avsq] = pcaApply(X_certain', U, mu, 2000 );
size(scores)
%%
scores_train_certain = scores(:,1:3642)';
%scores_train_certain = scores_train(logical(certain_train), :);
toc
%
tic
[accuracy, Ypredicted, Ytest] = cross_validation(scores_train_certain(1:3000,:), Y, 5, @svm_predict);
mean(accuracy)
toc

%% other features:

% Detect SURF features
% train_surf = [];
min_eigen = [];
haar_corners = [];
for i = 1:size(train_grey,3)
    i
face = train_grey(:,:,70);
% ftrs = detectSURFFeatures(face);

me = detectMinEigenFeatures(face);
me=me.selectStrongest(20);
me_feat = [me.Location, me.Metric];
me_feat = reshape(me_feat, [1 60]);
min_eigen = [min_eigen; me_feat];

haar  = detectHarrisFeatures(face);
haar = haar.selectStrongest(20);
haar_feat = [haar.Location haar.Metric];
haar_feat = reshape(haar_feat, [1 size(haar_feat,1)*size(haar_feat,2)]);
haar_corners = [haar_corners; haar_feat];

% regions = detectMSERFeatures(face);
% figure; imshow(face); hold on;
% plot(regions, 'showPixelList', true, 'showEllipses', false);
% region_feat = [regions.Location regions.Axes regions.Orietation];
% mser = [mser; region_feat];
% imshow(face); hold on;
% plot(corners.selectStrongest(20));
%Plot facial features.
% imshow(face);hold on; plot(ftrs);
% ftrs = ftrs.selectStrongest(3);
% surfvector = [ftrs.Location ftrs.Scale ftrs.Metric ftrs.SignOfLaplacian ftrs.Orientation];
% surfvector = reshape(surfvector,[18 1]);
% train_surf = [train_surf;surfvector'];
end

%% 
% load('face_certain.mat', 'certain');
% load('train_hog.mat', 'train_hog');
% load('test_hog.mat', 'test_hog');
tic
acc = [];
% for i = 1:50
% pc = 1:170;
certain_train = certain(1:5000);
X = train_hog;
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

%% Face Alignment
addpath ./face-release1.0-basic
% load('train_grey_faces.mat', 'train_grey');
% load('test_grey_faces.mat', 'test_grey');


load face_p146_small.mat
% 5 levels for each octave
model.interval = 5;
% set up the threshold
model.thresh = min(-0.65, model.thresh);


% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13 
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end

tic
im = train_grey(:,:,100);
im2 = cat(3, im, im);
im3 = cat(3, im, im2);
size(im3)
% im = reshape(im, [100 100 3]);
bs = detect(im3, model, model.thresh);
bs = clipboxes(im3, bs);
% insertObjectAnnotation(im3,'rectangle',bs.,'Face');
bs = nms_face(bs,0.3);
toc
figure,showboxes(im3, bs(1),posemap),title('Highest scoring detection');
