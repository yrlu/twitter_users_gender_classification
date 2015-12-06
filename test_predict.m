% Author: Max Lu
% Date: Dec 5

% prepare data:



% 
% 
% tic
% clear
% disp('Loading data..');
% load('train/genders_train.mat', 'genders_train');
% addpath('./liblinear');
% addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
% addpath('./libsvm');
% toc
% 
% disp('Preparing data..');
% 
% % Load image data from .mat
% % clear
% disp('img features...')
% tic
% load('train/genders_train.mat', 'genders_train');
% load('train/images_train.mat', 'images_train');
% load('test/images_test.mat', 'images_test');
% 
% % Set variables: train_grey and test_grey are the gray-scale images 
% % we use to detect faces on and to do feature extractions. 
% % Note: we used the first 2 observations twice to make partition of 
% % cross-validation easier. This seems to have little impact on the
% % classifier. 
% train_x = [images_train; images_train(1,:); images_train(2,:)];
% test_x =  images_test;
% 
% [train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
% [test_r, test_g, test_b, test_grey] = convert_to_img(test_x);
% grey_imgs = cat(3, train_grey, test_grey);
% red_imgs = cat(3, train_r, test_r);
% green_imgs = cat(3, train_g, test_r);
% blue_imgs = cat(3, train_b, test_b);
% 
% n_train_grey = size(train_grey,3);
% n_test_grey = size(test_grey,3);
%  n_total = n_train_grey + n_test_grey;
% % Detect and crop faces, eyes, noses from images, 
% % then extract HOG features on them. 
% % Preallocate arrays to store extracted HOG features
% face_hog = zeros(n_total, 5400);
% nose_hog = zeros(n_total, 900);
% eyes_hog = zeros(n_total, 792);
% % Create cascade detector objects for face, nose and eyes.
% faceDetector = vision.CascadeObjectDetector();
% NoseDetect = vision.CascadeObjectDetector('Nose');
% EyeDetect = vision.CascadeObjectDetector('EyePairSmall');
% % Create a binary vector to index the face-detected images 
% certain = ones(n_total,1);
% % Loop through all gray images
% for i  = 1:n_total
%     i
%     profile = grey_imgs(:,:,i);
%     bbox  = step(faceDetector, profile);
%     if ~isempty(bbox) % if any faces detected, get the first one
%         profile = imcrop(profile,bbox(1,:));
%         profile=imresize(profile,[100 100]);
%         grey_imgs(:,:,i) = profile;
%     else 
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
%         else 
%         certain(i) = 0;
%        end
%     end
%     img = grey_imgs(:,:, i); % extract HOGs multiple times
%     [featureVector, ~] = extractHOGFeatures(img);
%     img = imgaussfilt(img);
%     img = imresize(img, [50 50]);
%     [featureVector2, ~] = extractHOGFeatures(img);
%     img = imgaussfilt(img);
%     img = imresize(img, [25 25]);
%     [featureVector3, ~] = extractHOGFeatures(img);
%     face_hog(i,:) = [featureVector featureVector2 featureVector3];
%     profile = grey_imgs(:,:, i);
%     bbox_nose = step(NoseDetect, profile);
%     if ~isempty(bbox_nose) % if any nose detected, get the first one
%         nose = imcrop(profile,bbox_nose(1,:));
%         nose = imresize(nose,[50 50]);
%         [nose_hog(i,:),~] = extractHOGFeatures(nose);
%     end
%     bbox_eye = step(EyeDetect, profile);
%     if ~isempty(bbox_eye) % if any pair of eyes detected, get the first one
%         eyes = imcrop(profile,bbox_eye(1,:));
%         eyes=imresize(eyes,[25 100]);
%         [eyes_hog(i,:), ~] = extractHOGFeatures(eyes);
%     end
% end
% toc
% 
% % hog_feat = [face_hog nose_hog eyes_hog];
% % hog_feat_certain = hog_feat(logical(certain),:);
% % [U mu vars] = pca_1(hog_feat_certain');
% 
% 
% % save certain_HOG.mat eyes_hog face_hog nose_hog certain U mu vars

%%
clear all;
tic
disp('Add paths ...');
addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');
toc


disp('Preparing image features ...');

load('test/images_test.mat', 'images_test');

% Set variables: train_grey and test_grey are the gray-scale images 
% we use to detect faces on and to do feature extractions. 
% Note: we used the first 2 observations twice to make partition of 
% cross-validation easier. This seems to have little impact on the
% classifier. 
% train_x = [images_train; images_train(1,:); images_train(2,:)];
test_x =  images_test;

% [train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);
% grey_imgs = cat(3, train_grey, test_grey);
% red_imgs = cat(3, train_r, test_r);
% green_imgs = cat(3, train_g, test_r);
% blue_imgs = cat(3, train_b, test_b);

grey_imgs = test_grey;
red_imgs = test_r;
green_imgs = test_g;
blue_imgs = test_b;

% n_train_grey = size(train_grey,3);
n_test_grey = size(test_grey,3);
%  n_total = n_train_grey + n_test_grey;
 n_total = n_test_grey;
% Detect and crop faces, eyes, noses from images, 
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
    i
    profile = grey_imgs(:,:,i);
    bbox  = step(faceDetector, profile);
    if ~isempty(bbox) % if any faces detected, get the first one
        profile = imcrop(profile,bbox(1,:));
        profile=imresize(profile,[100 100]);
        grey_imgs(:,:,i) = profile;
    else 
        bbox_r  = step(faceDetector, red_imgs(:,:, i));
        bbox_g  = step(faceDetector, green_imgs(:,:, i));
        bbox_b = step(faceDetector, blue_imgs(:,:, i));
        if ~isempty(bbox_r)
            profile = imcrop(profile,bbox_r(1,:));
            profile=imresize(profile,[100 100]);
            grey_imgs(:,:, i) = profile;
        elseif ~isempty(bbox_g)
            profile = imcrop(profile,bbox_g(1,:));
            profile=imresize(profile,[100 100]);
            grey_imgs(:,:, i) = profile;
        elseif ~isempty(bbox_b)
            profile = imcrop(profile,bbox_b(1,:));
            profile=imresize(profile,[100 100]);
            grey_imgs(:,:, i) = profile;        
        else 
        certain(i) = 0;
       end
    end
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

load('models/submission/U_mu_vars.mat', 'U', 'mu');
% load('certain_HOG.mat', 'U', 'mu');


certain_test = certain;
hog_feat = [face_hog nose_hog eyes_hog];
[pca_hog,Xhat,avsq] = pcaApply(hog_feat', U, mu, 1500);
pca_hog = double(pca_hog');
img_test_x = pca_hog;

% save('U_mu_vars.mat', 'U', 'mu','vars');


tic

load('train/genders_train.mat', 'genders_train');
% Separate the data into training set and testing set.
Y = [genders_train; genders_train(1); genders_train(2,:)];
train_y = Y;

[words_train_X, words_test_X] = gen_data_words();
[image_features_train, image_features_test] = gen_data_image_features();
words_train_x = words_train_X;
words_test_x = words_test_X;
test_y = ones(size(words_test_x,1),1);



% [train_fs, test_fs] = gen_data_words_imgfeat_fs(1000);
% train_x_fs = train_fs;
% test_x_fs = test_fs;
% train_y_fs = Y;


% cols_sel=index(1:Nfeatures);
load('./models/submission/top_data_index.mat' , 'cols_sel');
words_train_s = [words_train_X, image_features_train];
words_test_s = [words_test_X, image_features_test];
train_x_fs = words_train_s(:, cols_sel);
test_x_fs = words_test_s(:, cols_sel);




toc

disp('Loading models..');
% load models:
load('models/submission/log_ensemble.mat','LogRens');
load('models/submission/log_model.mat', 'log_model');
load('models/submission/logboost_model.mat','logboost_model');
load('models/submission/svm_kernel_n_model.mat', 'svm_kernel_n_model');
load('models/submission/svm_kernel_model.mat', 'svm_kernel_model');
load('models/submission/svm_hog_model.mat', 'svm_hog_model');
load('models/submission/nn.mat', 'nn');

mdl.LogRens= LogRens;
mdl.log_model = log_model;
mdl.logboost_model = logboost_model;
mdl.svm_kernel_n_model = svm_kernel_n_model;
mdl.svm_kernel_model = svm_kernel_model;
mdl.svm_hog_model = svm_hog_model;
mdl.nn =nn;

toc
% make prediction:
disp('Making predictions..');
[~, yhat_log] = a_logistic_predict(mdl.log_model,words_test_x);
[~, yhat_nn] = a_nn_predict(mdl.nn,words_test_x);
[~, yhat_fs] = a_ensemble_trees_predict(mdl.logboost_model, test_x_fs);
toc
[~, yhat_kernel_n] = a_predict_kernelsvm_n(mdl.svm_kernel_n_model, train_x_fs, test_x_fs);
[~, yhat_kernel] = a_predict_kernelsvm(mdl.svm_kernel_model, train_x_fs, test_x_fs);
toc
[yhog, yhat_hog] = a_svm_hog_predict(mdl.svm_hog_model, img_test_x);

yhat_hog(logical(~certain_test),:) = 0;





ypred2 = [yhat_log yhat_fs yhat_nn yhat_hog];
% ypred2 = [yhat_log yhat_fs yhat_hog];
ypred2 = sigmf(ypred2, [2 0]);
yhat_kernel_n = sigmf(yhat_kernel_n, [1.5 0]);
yhat_kernel = sigmf(yhat_kernel, [1.5 0]);
ypred2 = [ypred2 yhat_kernel_n yhat_kernel];


Yhat = predict(test_y, sparse(ypred2), mdl.LogRens, ['-q', 'col']);
disp('Done!');
toc