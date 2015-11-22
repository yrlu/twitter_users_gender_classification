% Author: Max Lu
% Date: Nov 20


%% load data first ..

% Load the data first, see prepare_data.
if exist('genders_train','var')~= 1
prepare_data;
load('train/genders_train.mat', 'genders_train');
load('train/images_train.mat', 'images_train');
load('train/image_features_train.mat', 'image_features_train');
load('train/words_train.mat', 'words_train');
load('test/images_test.mat', 'images_test');
load('test/image_features_test.mat', 'image_features_test');
load('test/words_test.mat', 'words_test');
end

%%

train_x = [images_train; images_train(1,:); images_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2)];
test_x =  images_test;

[train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);

%%

train_test = [images_train; images_train(1,:); images_train(2,:); images_test];
[r, g, b, grey] = convert_to_img(train_test);
X = grey; 
[h w n] = size(grey);
x = reshape(X,[h*w n]); 
[img_coef, img_scores, img_eigens] = pca(x');
save('img_coef.mat', 'img_coef');
save('img_scores.mat', 'img_scores');
save('img_eigens.mat', 'img_eigens');

%% visualization


for i = 1:16
subplot(4,4,i) 
imagesc(reshape(img_coef(:,i),h,w)) 
end


%% 


% acc = [];
% for i = 1:50

X = img_scores(1:5000, 1:100);
Y = [genders_train; genders_train(1);genders_train(2)];
addpath('./liblinear');

% B = TreeBagger(95,X,Y, 'Method', 'classification');
% RFpredict = @(train_x, train_y, test_x) sign(str2double(B.predict(test_x)) - 0.5);

[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @rand_forest);
accuracy
mean(accuracy)
toc

% acc = [acc; mean(accuracy)];
% end

% plot(acc);

%%


train_scores = img_scores(:, 1: 5000);
test_scores = img_scores(:, 5001:end);


colormap gray 
for i = 1:64
subplot(8,8,i) 
imagesc(reshape(img_coef(:,i),h,w)) 
end


%%
tic
[h,w,n] = size(train_grey); 
d = h*w; 
% vectorize images 
x = reshape(train_grey,[d n]); 
x = double(x); 
%subtract mean 
x=bsxfun(@minus, x', mean(x'))'; 
% calculate covariance 
s = cov(x'); 
% obtain eigenvalue & eigenvector 
[V,D] = eig(s);
eigval = diag(D); 
% sort eigenvalues in descending order 
eigval = eigval(end:-1:1); 
V = fliplr(V); 
% show 0th through 15th principal eigenvectors 
eig0 = reshape(mean(x,2), [h,w]); 
figure,subplot(4,4,1) 
imagesc(eig0) 
colormap gray 
for i = 1:63
subplot(8,8,i+1) 
imagesc(reshape(V(:,i),h,w)) 
end
toc