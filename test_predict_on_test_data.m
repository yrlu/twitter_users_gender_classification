%% test_predict modified for the NO-TRAINd data case

% Author: Max Lu
% Date: Dec 5

% prepare data:
addpath('./liblinear');
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./libsvm');

% Separate the data into training set and testing set.
% Y = [genders_train; genders_train(1); genders_train(2,:)];
% train_y = Y;
% load('test/words_test.mat', 'words_test');
 % Inside make_final_prediction.m we only have X_test and X_train  
 % This is our preloaded format
 % X_test = [word_test img_feat_test img_test];
 % X_train = [words_train image_features_train genders_train]; %Note we will need Y_train 

 words_train = Xtrain(:, 1:5000);
 words_test = Xtest(:, 1:5000);
 
 words_train_x = [words_train; words_train(1,:); words_train(2,:)];
 words_test_x = words_test;
% [words_train_X, words_test_X] = gen_data_words();
% words_train_x = words_train_X;
% dwords_test_x = words_test_X;
 test_y = ones(size(words_test_x,1),1);
% test_y = Y(idx);

%% 
 img_feature_train = Xtrain(:,5001:5007);
 img_feature_test = Xtest(:,5001:5007); 
 images_test = Xtest(:,5008:35007);
 

% Assume this part was run after the image_feature_extract.m
% Here I copy the code
test_x =  images_test;
[red_imgs, green_imgs, blue_imgs, grey_imgs] = convert_to_img(test_x);
n_total = size(grey_imgs,3);

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
 %load('certain_HOG.mat', 'eyes_hog', 'face_hog','nose_hog', 'certain');
%    hog_feat = [face_hog nose_hog face_hog];
%    hog_feat_certain = hog_feat(logical(certain),:);
    load('U_mu_vars.mat', 'U', 'mu','vars');
    
    hog_feat = [face_hog nose_hog eyes_hog];
    hog_feat_certain = hog_feat(logical(certain),:);

    % [U mu vars] = pca_1(hog_feat_certain');
    [pca_hog,Xhat,avsq] = pcaApply(hog_feat', U, mu, 1500);
    pca_hog = double(pca_hog');
    
   
   % [~,certain,pca_hog] = gen_data_hog();
   % certain_train = certain(1:5000,:);
    certain_test = certain;

%     img_train_y_certain = Y(logical(certain_train), :);
% 
% img_train = pca_hog(1:5000,:);
% img_train_x_certain = img_train(logical(certain_train), :);
% img_train_x = img_train;
    img_test_x = pca_hog;


% % Features selection 
% [train_fs, test_fs] = gen_data_words_imgfeat_fs(1000);
% [top_igs, index]=sort(IG,'descend');
load top_data_index.mat cols_sel
% cols_sel=index(1:Nfeatures);
words_train_s = [words_train_x, img_feature_train];
words_test_s = [words_test img_feature_test];
train_x_fs = words_train_s(:, cols_sel);
test_x_fs = words_test_s(:, cols_sel);
% train_x_fs = train_fs;
% test_x_fs = test_fs;
%train_y_fs = Y;

toc

disp('Loading models..');
% load models:


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
% [ylbp, yhat_lbp, svm_lbp_model] = svm_predict(img_lbp_train_x_certain,img_train_y_certain, img_lbp_test_x, test_y);
yhat_hog(logical(~certain_test),:) = 0;
% yhat_lbp(logical(~certain_test),:) = 0;




ypred2 = [yhat_log yhat_fs yhat_nn yhat_hog];
ypred2 = sigmf(ypred2, [2 0]);
yhat_kernel_n = sigmf(yhat_kernel_n, [1.5 0]);
yhat_kernel = sigmf(yhat_kernel, [1.5 0]);
ypred2 = [ypred2 yhat_kernel_n yhat_kernel];


Yhat = predict(test_y, sparse(ypred2), mdl.LogRens, ['-q', 'col']);
disp('Done!');
toc