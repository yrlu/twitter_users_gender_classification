%% analysis PCA recontruction on train, test, train+test
% Deng, Xiang 11/19
clear all
close all
%%
% 1----on image_features_train
load('data\image_features_train.mat')
X_full=image_features_train;
X_mean=mean(X_full);
%A = [1 2 10; 1 4 20;1 6 15] ;
%C = bsxfun(@minus, A, mean(A))
% above is deviation from mean example
[coeff, score, latent]=pca(X_full);

accuracy=zeros(size(coeff,1),1);
X_dev=bsxfun(@minus, X_full, mean(X_full));
for i=1:size(coeff,1)
    Xp=score(:,1:i)*coeff(:,1:i)';
    err_orig=norm(X_dev,'fro');
    Xp_dev=Xp-X_dev; %(X-(Xp+mean(X))=(Xdev-Xp)
    err=norm(Xp_dev,'fro');
    accuracy(i)=1-(err^2/err_orig^2);
end
%
figure
plot(1:size(coeff,1),accuracy)
title ('recontruction accuracy vs number of principle components')
xlabel('PC#')
ylabel('Accuracy')
hold on
grid on
%%
clear all
load('data\image_features_test.mat')
% 2----on image_features_test
X_full=image_features_test;
X_mean=mean(X_full);
%A = [1 2 10; 1 4 20;1 6 15] ;
%C = bsxfun(@minus, A, mean(A))
% above is deviation from mean example
[coeff, score, latent]=pca(X_full);

accuracy=zeros(size(coeff,1),1);
X_dev=bsxfun(@minus, X_full, mean(X_full));
for i=1:size(coeff,1)
    Xp=score(:,1:i)*coeff(:,1:i)';
    err_orig=norm(X_dev,'fro');
    Xp_dev=Xp-X_dev; %(X-(Xp+mean(X))=(Xdev-Xp)
    err=norm(Xp_dev,'fro');
    accuracy(i)=1-(err^2/err_orig^2);
end
%
%figure
plot(1:size(coeff,1),accuracy)
%% combine
clear all

load('data\image_features_test.mat')
load('data\image_features_train.mat')
X_full=[image_features_train;image_features_test];
X_mean=mean(X_full);
%A = [1 2 10; 1 4 20;1 6 15] ;
%C = bsxfun(@minus, A, mean(A))
% above is deviation from mean example
[coeff, score, latent]=pca(X_full);

accuracy=zeros(size(coeff,1),1);
X_dev=bsxfun(@minus, X_full, mean(X_full));
for i=1:size(coeff,1)
    Xp=score(:,1:i)*coeff(:,1:i)';
    err_orig=norm(X_dev,'fro');
    Xp_dev=Xp-X_dev; %(X-(Xp+mean(X))=(Xdev-Xp)
    err=norm(Xp_dev,'fro');
    accuracy(i)=1-(err^2/err_orig^2);
end
%
%figure
plot(1:size(coeff,1),accuracy)
legend('train','test','combined')
hold off