%% landmark Hog test
load landmark_test_points.mat IsT bboxesT p 
%% 
% note: the for each row, p contains 29 points stored as (i,i+29)
% ppt1 = p(1,:);
% ptof1 = zeros(29,2);
% ptof1(:,1) = ppt1(1:29);
% ptof1(:,2) = ppt1(30:58);


%% hogs -> hogs around the landmarks 29*36
n = size(IsT,1);
hogs = zeros(n,1044);
for i = 1:n 
ppt = p(i,:);
ptof = zeros(29,2);
ptof(:,1) = ppt(1:29);
ptof(:,2) = ppt(30:58);
[hog, pts, vis]= extractHOGFeatures(IsT{i},ptof);
% imshow(IsT{1});
% for i = 1:29
%     hold on;
%     plot(ptof1(i,1),ptof1(i,2),'r.','MarkerSize',20);
% end
% 
% plot(vis) 
hogs(i,1:1044) = imresize(hog, [1 1044]);
end

% save 'hogs_landmarks' 'hogs'
%% 
train_hogs_full = zeros(4997, 5400);
test_hog_vis = [];
for i = 1:size(test_grey,3)
i
img = test_grey(:,:,i);
[featureVector, ~] = extractHOGFeatures(img);


img = imgaussfilt(img);
img = imresize(img, [50 50]);

[featureVector2, ~] = extractHOGFeatures(img);

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
%% 
train_hogs_full = zeros(n, 4356);
for i = 1:n 
[hogFull, ~] = extractHOGFeatures(IsT{i});
train_hogs_full(i,:) = hogFull;
end
% save hogs_full_certain train_hogs_full
%%
addpath('./train');
load genders_train
%
%%
X = [train_hogs_full hogs];
Y = genders_train(logical(certain));
% Y = train_y;
addpath('./liblinear');

% B = TreeBagger(95,X,Y, 'Method', 'classification');
% RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @svm_predict);
i
accuracy
mean(accuracy)
% end
%%
addpath('./liblinear');
[accuracy, Ypredicted, Ytest] = cross_validation(train_hogs_full, Y, 5, @logistic);
i
accuracy
mean(accuracy)
