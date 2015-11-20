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



parpool('local',1)
countTests = 200;
performances = zeros(1, countTests);
% [x, t] = house_dataset;
% x = X(1:n/2,:);
% t = Y(1:n/2);
% y = Y(n/2+1:n)
x = sparse(X(1:n,:));
t = Y(1:n);

tStart = tic;
for i = 1 : countTests,
    net = fitnet(10);
    net.trainFcn = 'trainscg';
    net.trainParam.showWindow = false;
    net = train(net, x', t', 'useParallel', 'yes', 'useGPU', 'only');
    y = net(x');
    per = perform(net, t', y);
    performances(i) = per;
end
elapsedTime = toc(tStart);
parpool close

display(sprintf('Average performance: %.1f', mean(performances)));
display(sprintf('Elapsed time: %.1f seconds', elapsedTime));