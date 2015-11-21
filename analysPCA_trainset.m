%% analysis PCA recontruction on train, test, train+test
% Deng, Xiang  since 11/19-
clear all
close all
%%
% 1----on image_features_train
load('.\data\image_features_train.mat')
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
male_idx=find(genders_train==0);
female_idx=find(genders_train==1);
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
%% Plot PCA coordinates male- female
figure
for i=1:7
    subplot(3,3,i)
    bar([coeff_male(:,i),coeff_female(:,i)]);
    title (['Coordinate of ' num2str(i) 'th PC-male/PC-female w.r.t the original feature space'])
    legend('PC-male','PC-female')
    hold off
    
end

% The first thing that surprises me is that each PC is dominated by a certain original image feature,
% as shown in the figure. Since the  PCs for a given dataset are linearly independent, does this imply
%that the original image features are very likely to be independent to each other?
% Furthermore, PC3-male is characterized by feature-1, PC4-male is dominated by feature-5; PC3-female
%is characterized by feature-5, PC4-female is dominated by feature-1--- it seems like for male, feature
%1 has more variance than feature 5, while for female they are flipped; and the other feature variance
%are in the same order. How to best use of this information?

%% Validate good PCA on smaller set
close all
X_full=X_male_train(900:1000,:);
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

X_full=X_female_train(900:1000,:);
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
figure
for i=1:7
    subplot(3,3,i)
    histogram(X_male_train(1:2000,i),20); %trunc 1:2000 since they are not the same size
    hold on
    histogram(X_female_train(1:2000,i),20);
    title (['Image' num2str(i) 'th feature intersection, male vs female'])
    hold off
end

%% Now, see if running PCA seperately will help
% run PCA on male
male_idx=find(genders_train==0);
female_idx=find(genders_train==1);
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
% Now project male onto PCA-global, project female onto PCA-global
figure
for i=1:7
    x_male_proj=X_male_train*coeff(:,i);
    x_female_proj=X_female_train*coeff(:,i);
    
    subplot(3,3,i)
    histogram(x_male_proj(1:2000,:),30);
    hold on
    histogram(x_female_proj(1:2000,:),30);
    title (['Image feature male, female projected on the' num2str(i) 'th global PCA']);
    legend('M','F')
    hold off
    
    
end
% Now project male onto PCA-male, project female onto PCA-female
figure
for i=1:7
    x_male_proj=X_male_train*coeff_male(:,i);
    x_female_proj=X_female_train*coeff_female(:,i);
    
    subplot(3,3,i)
    histogram(x_male_proj(1:2000,:),30);
    hold on
    histogram(x_female_proj(1:2000,:),30);
    title (['Image feature male, female projected on the' num2str(i) 'th male/female PCA']);
    legend('M','F')
    hold off
    
    
    
end

% see what the test set projection looks like

figure
for i=1:7
    x_test_proj_male=image_features_test*coeff_male(:,i);
    x_test_proj_female=image_features_test*coeff_female(:,i);
    
    subplot(3,3,i)
    histogram(x_test_proj_male(:,:),30);
    hold on
    histogram(x_test_proj_female(:,:),30);
    title (['Image feature test set projected on the' num2str(i) 'th male/female PCA']);
    legend('M','F')
    hold off
    
end


%% The experiments above indicate each indiviual classifer might be a weak classifier
% but observe that PC3-male always --- PC4-female, this is weird! This
% probbly indicates that all the features whole together might be a strong
% classifier! And this definitely doesn't meem PCA is useless.
close all
male_idx=find(genders_train==0);
female_idx=find(genders_train==1);
X_male_train=image_features_train(male_idx,:);
X_female_train=image_features_train(female_idx,:);
figure
plot(X_male_train(1:200,1),X_male_train(1:200,2),'r+')
hold on
plot(X_female_train(1:200,1),X_female_train(1:200,2),'bo')
hold off

figure
plot(X_male_train(1:200,3),X_male_train(1:200,2),'r+')
hold on
plot(X_female_train(1:200,3),X_female_train(1:200,2),'bo')
hold off

figure
plot(X_male_train(1:200,5),X_male_train(1:200,2),'r+')
hold on
plot(X_female_train(1:200,5),X_female_train(1:200,2),'bo')
hold off
%% two features doesn't work really well, now look at intersection
close all
male_idx=find(genders_train==0);
female_idx=find(genders_train==1);
X_male_train=image_features_train(male_idx,:);
X_female_train=image_features_train(female_idx,:);
Np=2000;
intersection=zeros(Np,7);
for i=1:Np
    intersection(i,:)=min([X_male_train(i,:);X_female_train(i,:)]);
end
figure
for i=1:7
    subplot(3,3,i)
    
    
    histogram(X_male_train(1:Np,i),20); %trunc 1:2000 since they are not the same size
    hold on
    histogram(X_female_train(1:Np,i),20);
    hold on
    histogram( intersection(:,i),20);
    
    title (['Image feature male-female intersection'])
    legend('male','female','intersection')
    hold off
end

%% Interaction between features
mean_train_male=mean(X_male_train);
bin_X_male_train=X_male_train;
for i=1:7
    bin_X_male_train(:,i)=bin_X_male_train(:,i)/ mean_train_male(i);
end
Inter_X_male_train=bin_X_male_train'*bin_X_male_train;
%Inter_X_male_train=Inter_X_male_train/det(Inter_X_male_train);
figure
imagesc(Inter_X_male_train);
%colormap('gray')

mean_train_female=mean(X_female_train);
bin_X_female_train=X_female_train;
for i=1:7
    bin_X_female_train(:,i)=bin_X_female_train(:,i)/ mean_train_female(i);
end
Inter_X_female_train=bin_X_female_train'*bin_X_female_train;
%Inter_X_female_train=Inter_X_female_train/det(Inter_X_female_train);
figure
imagesc(Inter_X_female_train);
%colormap('gray')

%% Since by PCA or interection the features seems like to be independent, now try NB
Nc=3000;
cols_sel=[1 2 5 7];
X_train_split_train=image_features_train(1:Nc,cols_sel);
X_train_split_train_labels=genders_train(1:Nc);
X_train_split_test=image_features_train(Nc+1:end,cols_sel);
X_train_split_test_labels=genders_train(Nc+1:end);
nb_train = NaiveBayes.fit(X_train_split_train , X_train_split_train_labels);
cpre = nb_train.predict(X_train_split_test);
err=sum(X_train_split_test_labels ~= cpre)/size(X_train_split_test_labels,1);%compute error
accuracy_orig=1-err

%% SVM
Nc=3000;
cols_sel=[1 2 5 7];
X_train_split_train=image_features_train(1:Nc,cols_sel);
X_train_split_train_labels=genders_train(1:Nc);
X_train_split_test=image_features_train(Nc+1:end,cols_sel);
X_train_split_test_labels=genders_train(Nc+1:end);
model = fitcsvm(X_train_split_train , X_train_split_train_labels);
cpre = model.predict(X_train_split_test);
err=sum(X_train_split_test_labels ~= cpre)/size(X_train_split_test_labels,1);%compute error
accuracy_orig=1-err

%% Kmeans
Nc=3000;
cols_sel=[1];
% binarize...
bin_image_features_train=image_features_train;
mean_X=mean(image_features_train);
for i=1:7
    bin_image_features_train(:,i)=bin_image_features_train(:,i)/mean_X(i);
end

X_train_split_train=bin_image_features_train(1:Nc,cols_sel);
X_train_split_train_labels=genders_train(1:Nc);
X_train_split_test=bin_image_features_train(Nc+1:end,cols_sel);
X_train_split_test_labels=genders_train(Nc+1:end);
[indices, Ctrs]    = kmeans(X_train_split_train, 2,'MaxIter',500);
clusterLabels=zeros(2,1);
for k=1:2
    cur_ids=find(indices==k);
    clusterLabels(k)=mode(X_train_split_train_labels(cur_ids)); % label cluster with most frequent letter
end
cpre=zeros(size(X_train_split_test_labels,1),1);
for j=1:size(X_train_split_test,1)
    distances=zeros(size(Ctrs,1),1);
    for k=1:size(Ctrs,1)
        distances(k)=sum((Ctrs(k,:)-X_train_split_test(j,:)).^2);
    end
    [~,ci]=min(distances); %ci--cluster assignment
    cpre(j)=clusterLabels(ci);
end

err=sum(X_train_split_test_labels ~= cpre)/size(cpre,1);
accuracy=1-err

%% Random forest
Nc=3000;
cols_sel=[1:7];
NumTrees=70;
X_train_split_train=image_features_train(1:Nc,cols_sel);
X_train_split_train_labels=genders_train(1:Nc); 
X_train_split_test=image_features_train(Nc+1:end,cols_sel);
X_train_split_test_labels=genders_train(Nc+1:end);
model= TreeBagger(NumTrees,X_train_split_train,X_train_split_train_labels,'OOBPred','On');
cpre = str2double(model.predict(X_train_split_test));
err=sum(X_train_split_test_labels ~= cpre)/size(X_train_split_test_labels,1);%compute error
accuracy_orig=1-err