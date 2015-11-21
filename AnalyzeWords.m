%Deng, Xiang
% 11 20-
clear all
close all
load data\words_train.mat;
load ('data\genders_train.mat');
occupany= sum(sum(words_train~=0))/(4998*5000)
% This is NOT a sparse matrix!!!

%% run PCA recontruction
X_full=words_train;
X_mean=mean(X_full);
%A = [1 2 10; 1 4 20;1 6 15] ;
%C = bsxfun(@minus, A, mean(A))
% above is deviation from mean example
[coeff, score, latent]=pca(X_full);

accuracy=zeros(size(coeff,1),1);
X_dev=bsxfun(@minus, X_full, mean(X_full));
for i=400:450
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
male_idx=find(genders_train==1);
female_idx=find(genders_train==0);
X_male_train=words_train(male_idx,:);
X_female_train=words_train(female_idx,:); 
[coeff_male, score_male, latent_male]=pca(X_male_train);
[coeff_female, score_female, latent_female]=pca(X_female_train);
