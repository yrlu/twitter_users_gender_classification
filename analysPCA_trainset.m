%% analysis PCA recontruction on train, test, train+test
% Deng, Xiang  since 11/19-
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

%% split male female 
clear all 
close all
load ('data\genders_train.mat');
load('data\image_features_test.mat')
load('data\image_features_train.mat')
male_idx=find(genders_train==1);
female_idx=find(genders_train==0);
%n male=2840; barely evenly split
X_male_train=image_features_train(male_idx,:);
X_female_train=image_features_train(female_idx,:); 


% still agrees with above observation
X_full=X_male_train;
X_mean=mean(X_full);
%A = [1 2 10; 1 4 20;1 6 15] ;
%C = bsxfun(@minus, A, mean(A))
% above is deviation from mean example
[coeff, score, latent]=pca(X_full);
% top 2 PCA visualize...............
x_pca_1=score(:,1);
x_pca_2=score(:,2); 

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
 
grid on
%% find the most distriminative PCA between male and female
close all
X_full=X_male_train;
X_mean=mean(X_full);
[coeff_male, score_male, latent_male]=pca(X_full);
% top 2 PCA visualize...............
 
x_pca_1=score_male(:,7);
x_pca_2=score_male(:,2);
figure
plot(-x_pca_1,x_pca_2,'go')
hold on 
figure 
biplot(coeff_male(:,1:2),'Scores',score_male(:,1:2),'VarLabels',...
		{'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X7'})
 
X_full=X_female_train;
X_mean=mean(X_full);
[coeff_female, score_female, latent_female]=pca(X_full);
% top 2 PCA visualize...............
x_pca_1=score_female(:,7);
x_pca_2=score_female(:,2);
figure
plot(-x_pca_1,x_pca_2,'b+')
hold off
figure 
biplot(coeff_female(:,1:2),'Scores',score_female(:,1:2),'VarLabels',...
		{'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X7'})
     
PCA_corr=zeros(7,1);    
for i=1:size( coeff_male, 2)
    PCA_corr(i)=coeff_male(:,i)'*coeff_female(:,i); %% assuming there are 1-1 corrsponding, might be wrong...
 
end
PCA_good_idx=find(abs(PCA_corr)<0.5);

male_pca_i=score_male(:,3);
female_pca_i=score_female(:,3);
figure
histogram(male_pca_i,20)
hold on
histogram(female_pca_i,20)
hold off

male_pca_i=score_male(:,4);
female_pca_i=score_female(:,4);
figure
histogram(male_pca_i,20)
hold on
histogram(female_pca_i,20)
hold off

% HOLD ON!!!!!!!!!! THIS doesn't mean it must work! It actually seems like
% pc3 of male corresponds pc4 of female, not exactly 1-1 correspondence!
% We have to make sure pc3 of male always corresponds to pc4 of female,
% otherwise this method is not gauranteed to work! Uncomment below and
% check!
 %coeff_male(:,3)'*coeff_female(:,4);
 %coeff_male(:,4)'*coeff_female(:,3);
% male_pca_i=score_male(:,3);
% female_pca_i=score_female(:,4);
% figure
% histogram(male_pca_i,20)
% hold on
% histogram(female_pca_i,20)
% hold off
% male_pca_i=score_male(:,4);
% female_pca_i=score_female(:,3);
% figure
% histogram(male_pca_i,20)
% hold on
% histogram(female_pca_i,20)
% hold off
%% Validate good PCA on smaller set
close all
X_full=X_male_train(200:600,:);
X_mean=mean(X_full);
[coeff_male, score_male, latent_male]=pca(X_full);
% top 2 PCA visualize...............
 
x_pca_1=score_male(:,7);
x_pca_2=score_male(:,2);
figure
plot(-x_pca_1,x_pca_2,'go')
hold on 
figure 
biplot(coeff_male(:,1:2),'Scores',score_male(:,1:2),'VarLabels',...
		{'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X7'})
 
X_full=X_female_train(200:600,:);
X_mean=mean(X_full);
[coeff_female, score_female, latent_female]=pca(X_full);
% top 2 PCA visualize...............
x_pca_1=score_female(:,7);
x_pca_2=score_female(:,2);
figure
plot(-x_pca_1,x_pca_2,'b+')
hold off
figure 
biplot(coeff_female(:,1:2),'Scores',score_female(:,1:2),'VarLabels',...
		{'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X7'})
     
PCA_corr=zeros(7,1);    
for i=1:size( coeff_male, 2)
    PCA_corr(i)=coeff_male(:,i)'*coeff_female(:,i);%% assuming there are 1-1 corrsponding, might be wrong...
 
end
PCA_good_idx=find(abs(PCA_corr)<0.5);

%% Compare male femal image features
% it turns out the most discriminative feature is feature 1: age
close all
for i=1:7
    figure
    histogram(X_male_train(1:2000,i),20); %trunc 1:2000 since they are not the same size
    hold on
    histogram(X_female_train(1:2000,i),20); 
    title (['Image' num2str(i) 'th feature intersection, male vs female'])
     hold off
end

%% Now, see if running PCA seperately will help
% run PCA on male 
male_idx=find(genders_train==1);
female_idx=find(genders_train==0);
X_male_train=image_features_train(male_idx,:);
X_female_train=image_features_train(female_idx,:); 
[coeff_male, score_male, latent_male]=pca(X_male_train);
[coeff_female, score_female, latent_female]=pca(X_female_train);
%and recontruct male from their PCA
accuracy=zeros(size(coeff_male,1),1);
X_dev_male=bsxfun(@minus, X_male_train, mean(X_male_train));
for i=1:size(coeff_male,1)
    Xp=score_male(:,1:i)*coeff_male(:,1:i)';
    err_orig=norm(X_dev_male,'fro');
    Xp_dev=Xp-X_dev_male; %(X-(Xp+mean(X))=(Xdev-Xp)
    err=norm(Xp_dev,'fro');
    accuracy(i)=1-(err^2/err_orig^2);
end
%
figure
plot(1:size(coeff_male,1),accuracy) 
xlabel('PC#')
ylabel('Accuracy')
 
grid on
hold on
%recontruct male from global PCA
[coeff, score, latent]=pca(image_features_train);

accuracy=zeros(size(coeff,1),1);
X_dev=bsxfun(@minus, X_male_train, mean(X_male_train));
for i=1:size(coeff,1)
    Xp=score_male(:,1:i)*coeff(:,1:i)';
    err_orig=norm(X_dev,'fro');
    Xp_dev=Xp-X_dev; %(X-(Xp+mean(X))=(Xdev-Xp)
    err=norm(Xp_dev,'fro');
    accuracy(i)=1-(err^2/err_orig^2);
end
%
 
plot(1:size(coeff,1),accuracy)
title ('recontruction accuracy vs number of principle components')
xlabel('PC#')
ylabel('Accuracy')
legend('Recontruct male from PCA-male','Recontruct male from PCA-global');
hold  off
grid on