% proj_final README.TXT

Group: Woodpecker
Members: Xiang Deng, Yiren Lu, Dongni Wang
Date: Dec 5, 2015

The four methods we have implemented (and included here) are: 
Naive Bayes (generative method)
Logistic Regression; LogitBoost + Trees, ANN, SVM (discriminative method)
K-nearest Neighbors (instance-based method)
PCA (Semi-supervised dimensionality reduction of the data)

To test ALL, please check alg_demo.m

Besides these methods mentioned above, we have also tried GMM, K-means clustering, linear regression, random forest, perceptron, auto-encoder, fisherface, bagging, Adaboost and majority voting. Most of them does not perform very well (may be due to non-optimal configurations) and were discarded in our final model.   

%%%%%%%%%%%%%%%%%
%% Naive Bayes %%
%%%%%%%%%%%%%%%%%
The Naive Bayes model was implemented for text (words) classification. The first version is a multinomial document model, and the second and final version is a Bernoulli document model in which the presence and absence of words, instead of its frequency, are taken into account. 
This model was implemented based on ‘Text Classification using Naive Bayes’ by Hiroshi Shimodaira and various online tutorials. 
 
Training Data: words_train
Testing Data: words_test

Relevant files: (Inside /NB)
predict_MNNB.m
applyMNNB.m
trainMNNB.

To test: 
[testLabels, ~] = predict_MNNB(trainPoints, trainLabels, testPoints, ones(size(words_test,1),1));


%%%%%%%%%%%%%%%%%%%%%%%%%
%% Logistic Regression %%
%%%%%%%%%%%%%%%%%%%%%%%%%
The logistic Regression model was implemented for text and image classification. We used the ‘liblinear’ package written in C and chose the L2-regularized logistic regression. We also used Logistic Regression as an ensemble method (stacking). The raw (probability) outputs from other classifiers on the words or images features were the inputs of the final logistic regression ensembler.   

Training Data: words_train, labels/probability of training data predicted by other classifiers. 
Testing Data: words_test, labels/probability of testing data predicted by other classifiers. 


Relevant files: 
logistic.m
./liblinear

To test: 
[testLabels] = logistic(trainPoints, trainLabels, testPoints, ones(size(words_test,1),1));


%%%%%%%%%%%%%%%%%%%%%%%%
%% LogitBoost + Trees %%
%%%%%%%%%%%%%%%%%%%%%%%%
We built an ensemble predictor with LogitBoost Ensemble Method and 300 trees as weak learners. This ensemble was created using matlab’s built-in fitensemble function. We first used Information theory to rank the most informative word features. Then we experimented with multiple weak learner types, ensemble methods, and choices of number of ensemble members and number of word features to use. Cross-validation was used to select the best combination of parameters and settings. 

Training Data: words_train
Testing Data: words_test

Relevant files: 
acc_ensemble_trees.m

To test: 
[testLabels, ~] = acc_ensemble_trees(trainPoints, trainLabels, testPoints, ones(size(words_test,1),1));

%%%%%%%%%
%% SVM %%
%%%%%%%%%
The SVM was used with PCA-ed HOG features in classifying the images. We used the ‘libsvm’ package written in C and chose the cost to be ’10’ based on the results of cross-validation. 

Training Data: PCA-ed HOG features on cropped face images from the training set
Testing Data: PCA-ed HOG features on cropped face images from the testing set

Relevant files: 
svm_predict.m 
../libsvm
convert_to_img.m
pca_toolbox.m
pcaApply.m
test_face_detection.m

To Test: (see PCA for more details)
[testLabels,~] = svm_predict(Xtrain_pca, Ytrain, Xtest_pca, ones(size(words_test,1),1));


%%%%%%%%%
%% ANN %%
%%%%%%%%%
The ANN was employed for text classification. We chose two hidden layers with 100 and 50 nodes based on the results of cross-validation. The learning rate, batchsize and number of epochs, etc. are manually configured for this dataset. 

Training Data: words_train
Testing Data: words_test

Relevant files: 
acc_neural_net.m
../DL_toolbox

To Test: 
[testLabels] = acc_neural_net(trainPoints, trainLabels, testPoints, ones(size(words_test,1),1));


%%%%%%%%%%%%%%%%%%%%%%%%%
%% K-nearest Neighbors %%
%%%%%%%%%%%%%%%%%%%%%%%%%
The K-nearest Neighbors model was implemented for text classification. We used the KNN_TEST.m provided earlier this course in hw2_kit. To reduce the computational complexity and improve accuracy, we used information gain to select the most relevant features and experimented with various choices of numbers of nearest neighbors. The highest cross validation accuracy was achieved by using 16 neighbors, and top 70 most relevant words. 

Training Data: words_train
Testing Data: words_test

Relevant files: (Inside /KNN)
calc_information_gain.m
knn_test.m

To test: 
[testLabels] = knn_test(16, trainPoints, trainLabels, testPoints);


%%%%%%%%%%
%% PCA %%
%%%%%%%%%%
The PCA was implemented for image dimensionality reduction. We adapted the Piotr's Computer Vision Matlab Toolbox (Version 3.24). In the earlier stage, we applied the PCA directly to images and tested its performance with multiple classifiers. At last, we decided not to use PCA on images due to its poor performance, instead we applied PCA on extracted HOG features on face-detected images. Here, to simplify the demo process, we applied PCA on images and used SVM to obtain the labels on test set. For a better test performance (over 84%), we could crop the detected faces from images, extract HOG features on them, then use PCA and SVM. 

Training Data: images_train (gray-scale)
Testing Data: images_test (gray-scale)

Relevant files: (Inside /PCA)
convert_to_img.m
pca_toolbox.m
pcaApply.m
svm_predict.m 
test_face_detection.m
../libsvm

To test:
[~, ~, ~, train_grey] = convert_to_img(images_train);
[~, ~, ~, test_grey] = convert_to_img(images_test);
X = cat(3, train_grey, test_grey);
[h w n] = size(X);
x = reshape(X,[h*w n]); 

[[U,mu,vars] = pca_toolbox(x);
[scores,~,~] = pcaApply(x, U, mu, 2000);
YPC = double(scores’);
Xtrain_pca = YPC(1:size(train_grey,3),:);
Xtest_pca = YPC(size(train_grey,3)+1:end,:);
[testLabels,~] = svm_predict(Xtrain_pca, trainLabels, Xtest_pca, ones(size(words_test,1),1));

