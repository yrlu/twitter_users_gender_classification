% Load data
img_test = importdata('../test/images_test.txt');
img_feat_test = importdata('../test/image_features_test.txt');
word_test = importdata('../test/words_test.txt');
X_test = [word_test img_test img_feat_test];

model = init_model;
predictions = make_final_prediction(model, X_test);

% Use turnin on the output file
% turnin -c cis520 -p leaderboard submit.txt
dlmwrite('submit.txt', predictions);
