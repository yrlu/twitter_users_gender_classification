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

% Prepare/Load PCA-ed data,  
if exist('eigens','var')~= 1
    if exist('coef.mat','file') ~= 2 
        X = [words_train, image_features_train; words_test, image_features_test]; 
        [coef, scores, eigens] = pca(X);
        save('coef.mat', 'coef');
        save('scores.mat', 'scores');
        save('eigens.mat', 'eigens');
    else 
        load('coef.mat', 'coef');
        load('scores.mat', 'scores');
        load('eigens.mat', 'eigens');
    end
end



%%
[n m] = size(words_train);
X = [words_train, image_features_train; words_test, image_features_test];
Y = genders_train;

% PCA features
X = scores(1:n, 1:4900);
% X = X(1:n,:);

% add 2 more samples to make n = 5000;
X = [X;X(1,:);X(1,:)];
Y = [Y;Y(1,:);Y(1,:)];
% X = normc(X);

disp('Adaboost + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @adaboost);
accuracy
mean(accuracy)
toc
