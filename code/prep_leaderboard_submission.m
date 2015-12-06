% Load data
img_test = importdata('../test/images_test.txt');
img_feat_test = importdata('../test/image_features_test.txt');
word_test = importdata('../test/words_test.txt');

image_features_train = importdata('train/image_features_train.txt');
words_train = importdata('train/words_train.txt');
genders_train= importdata('train/genders_train.txt');

X_test = [word_test img_feat_test img_test];
X_train = [words_train image_features_train genders_train]; %Note we will need Y_train 

model = init_model();
predictions = make_final_prediction(model, X_test, X_train);

% Use turnin on the output file
% turnin -c cis520 -p leaderboard submit.txt
dlmwrite('submit.txt', predictions);
