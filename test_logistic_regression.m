% Author: Max Lu
% Date: Nov 20
% Nov 20 Update: use plain features 8 folds cross validation-> 86.63%

% add lib path:
addpath('./liblinear');

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
tic
% X = [words_train, image_features_train; words_test, image_features_test];
X = [words_train; words_test];
% X = normc(X);
Y = genders_train;
[n m] = size(words_train);

% PCA features
% X = scores(1:n, 1:3200);
X = X(1:n, :);
addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @logistic);
accuracy
mean(accuracy)
toc

%% Use new features from processed data. (Stemming and stopwords removal)

load('train/words_stem_train.mat', 'words_stem_train');
load('train/genders_train.mat', 'genders_train');
% load('test/words_stem_test.mat', 'words_stem_test');

% X = words_stem_train;
% X = words_train(:, logical(remove));
X = words_train;
Y = genders_train;

addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 5, @predict_MNNB);
accuracy
mean(accuracy)




%% try image features
% X = [words_train, image_features_train; words_test, image_features_test];
X = [images_train(:,1:10000); images_test(:,1:10000)];
% X = normc(X);
Y = genders_train;
[n m] = size(words_train);

% PCA features
% X = scores(1:n, 1:3200);
X = X(1:n, :);
addpath('./liblinear');
disp('logistic regression + cross-validation');
[accuracy, Ypredicted, Ytest] = cross_validation(X, Y, 8, @logistic);
accuracy
mean(accuracy)
toc


%%

train_x = words_train;
train_y = genders_train;
test_x = words_test;
test_y = ones(size(test_x,1), 1);

model = train(train_y, sparse(train_x), ['-s 0', 'col']);
[Yhat] = predict(test_y, sparse(test_x), model, ['-q', 'col']);
% dlmwrite('submit.txt',Yhat, '\n');