% Author: Max Lu
% Date: Dec 4

%%




load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');

%% Test Raw HOG pyramid + HOG nose + HOG eyes
% Logistic + Raw HOG => 81-82%
% SVM + Raw HOG => 83+% (c = 100)


load('train/genders_train.mat', 'genders_train');
load('train_hog_pry.mat', 'train_hog');
load('train_nose_hog.mat', 'train_nose_hog');
load('train_eyes_hog.mat', 'train_eyes_hog');
load('face_certain.mat', 'certain');

tic
train_x = [train_hog train_nose_hog train_eyes_hog];
train_y = [genders_train; genders_train(1); genders_train(2)];
certain_train = certain(1:5000);
X = double(train_x(logical(certain_train),:));
Y = train_y(logical(certain_train),:);

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @svm_predict_test);
accuracy
mean(accuracy)
toc

%% test svm + PCA over HOG features
% PCA1500 + SVM c = 10 => 84+%
% PCA1500 + SVM c = 100 => 81+%

tic
load('train_hog_pry.mat', 'train_hog');
load('train_nose_hog.mat', 'train_nose_hog');
load('train_eyes_hog.mat', 'train_eyes_hog');
load('face_certain.mat', 'certain');
load('img_pca_basis.mat', 'U', 'mu', 'vars');
certain_train = certain(1:5000);
X =  [train_hog train_nose_hog train_eyes_hog];
X_certain = X(logical(certain_train),:);
Y = train_y(logical(certain_train),:);

train_y = [genders_train; genders_train(1); genders_train(2)];
Y = train_y(logical(certain_train),:);

[YPC,Xhat,avsq] = pcaApply(X_certain', U, mu, 1500);
YPC = double(YPC');

[accuracy, Ypredicted, Ytest] = cross_validation(YPC, Y, 5, @svm_predict_test);
accuracy
mean(accuracy)
toc




%% 

tic
train_x = [images_train; images_train(1,:); images_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2)];
test_x =  images_test;

[train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);
toc

%% Face detection & cropping:
tic


img_grey = cat(3, train_grey, test_grey);
img_r = cat(3, train_r, test_r);
img_g = cat(3, train_g, test_g);
img_b = cat(3, train_b, test_b);
toc
%%
tic
faceDetector = vision.CascadeObjectDetector();
certain = ones(size(img_grey,3),1);

for i  = 1:size(img_grey,3)
profile = img_grey(:,:, i);
bbox            = step(faceDetector, profile);
if ~isempty(bbox)
i
bbox
profile = imcrop(profile,bbox(1,:));
profile=imresize(profile,[100 100]);
img_grey(:,:, i) = profile;
else
    bbox_r            = step(faceDetector, img_r(:,:, i));
    bbox_g            = step(faceDetector, img_g(:,:, i));
    bbox_b            = step(faceDetector, img_b(:,:, i));
    if ~isempty(bbox_r)
        profile = imcrop(profile,bbox_r(1,:));
        profile=imresize(profile,[100 100]);
        img_grey(:,:, i) = profile;
    elseif ~isempty(bbox_g)
        profile = imcrop(profile,bbox_g(1,:));
        profile=imresize(profile,[100 100]);
        img_grey(:,:, i) = profile;
    elseif ~isempty(bbox_b)
        profile = imcrop(profile,bbox_b(1,:));
        profile=imresize(profile,[100 100]);
        img_grey(:,:, i) = profile;
    else
        certain(i) = 0;
    end
end
end
% save('img_grey_faces.mat', 'img_grey');
% save('face_certain2.mat','certain');
toc



%% HOG feature extraction

hog_feat = zeros(9997, 28332);

for i = 1:size(img_grey,3)
    i
    img = img_grey(:,:,i);
    [featureVector1, hogVisualization1] = extractHOGFeatures(img,'CellSize',[20 20]);
    [featureVector2, hogVisualization2] = extractHOGFeatures(img,'CellSize',[16 16]);
    [featureVector3, hogVisualization3] = extractHOGFeatures(img,'CellSize',[12 12]);
    [featureVector4, hogVisualization4] = extractHOGFeatures(img,'CellSize',[8 8]);
    [featureVector5, hogVisualization5] = extractHOGFeatures(img,'CellSize',[4 4]);
    hog_feat(i,:) = [featureVector1,featureVector2,featureVector3,featureVector4,featureVector5];
end


%% LBP feature extraction
lbp_feat = zeros(9997, 15871);

for i = 1:size(img_grey,3)
    i
    img = img_grey(:,:,i);
    featureVector1 = extractLBPFeatures(img,'CellSize',[20 20]);
    featureVector2 = extractLBPFeatures(img,'CellSize',[16 16]);
    featureVector3 = extractLBPFeatures(img,'CellSize',[12 12]);
    featureVector4 = extractLBPFeatures(img,'CellSize',[8 8]);
%     featureVector5 = extractLBPFeatures(img,'CellSize',[4 4]);
    lbp_feat(i,:) = [featureVector1,featureVector2,featureVector3,featureVector4];
end



%% SURF features;
tic
surfpoints = zeros(1000, 7);
count = 1;
for i = 1:size(img_grey,3)
    i
    img = img_grey(:,:,i);
    points = detectSURFFeatures(img); 
    points = points.selectStrongest(1);
    for j = 1:size(points,1)
        p = points(j);
        surfpoints(count, :) = [p.Scale p.SignOfLaplacian p.Orientation p.Location p.Metric p.Count];
        count = count +1
        if count > 1000
            break;
        end
    end
    if count > 1000
        break;
    end
end
plot(100- surfpoints(:,4),100- surfpoints(:,5),'*');

numClusters = 50 ;
[means, covariances, priors] = vl_gmm(surfpoints', numClusters);
%%
addpath ./vlfeattoolbox/mex/mexmaci64/

surf_fv = zeros(size(img_grey,1),700);
for i = 1:size(img_grey,3)
   i
   img = img_grey(:,:,i);
   points = detectSURFFeatures(img); 
   points = points.selectStrongest(30);
   img_surf = zeros(size(points,1),7);
   for j = 1:size(points,1)
       p = points(j);
       img_surf(j,:) = [p.Scale p.SignOfLaplacian p.Orientation p.Location p.Metric p.Count];
   end
   
   encoding = vl_fisher(img_surf', means, covariances, priors);
   size(encoding)
   surf_fv(i,:) = encoding;
end

toc

%%
certain_train = certain(1:5000,:);
surf_fv_train = surf_fv(1:5000,:);
surf_fv_train_certain = surf_fv_train(logical(certain_train),:);
% surf_fv_train_certain = surf_fv_train_certain/norm(surf_fv_train_certain);
train_y = [genders_train; genders_train(1); genders_train(2)];
Y = train_y(logical(certain_train),:);
[accuracy, Ypredicted, Ytest] = cross_validation(surf_fv_train_certain, Y, 5, @svm_predict_test);
mean(accuracy)

%%
% load('train_hog_pry.mat', 'train_hog');

dopca =1;
addpath ./liblinear
tic
certain_train = certain(1:5000,:);
% train_x = double([train_hog train_nose_hog train_eyes_hog]);
% train_x = double([hog_feat(1:5000, :)]);
train_x = double([lbp_feat(1:5000,:)]);
% train_x = (train_x);
train_y = [genders_train; genders_train(1); genders_train(2)];
X = train_x(logical(certain_train),:);
Y = train_y(logical(certain_train),:);
% size(X)
% size(Y)

if dopca==1
[U mu vars] = pca_1(X');
end
[XPC,Xhat,avsq] = pcaApply(X', U, mu, 2000);
XPC = double(XPC');
toc
[accuracy, Ypredicted, Ytest] = cross_validation(XPC, Y, 5, @svm_predict_test);
accuracy
mean(accuracy)
toc




%% test fisher vector


addpath ./vlfeattoolbox/mex/mexmaci64/

numFeatures = 5000 ;
dimension = 2 ;
data = rand(dimension,numFeatures) ;
data = surfpoints(1:500,:)';

numClusters = 20 ;
[means, covariances, priors] = vl_gmm(data, numClusters);

numDataToBeEncoded = 1000;
dataToBeEncoded = rand(dimension,numDataToBeEncoded);
dataToBeEncoded =  surfpoints(501:550,:)';
encoding = vl_fisher(dataToBeEncoded, means, covariances, priors);
size(encoding)
% plot(encoding)