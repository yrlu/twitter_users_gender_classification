% D.Wang 11/23
% 
%% choose k for GMM
% PCA visualization: unlikely to work;
% Information criteria AIC or BIC 
% Examine AIC over varying numbers of components: 
% AIC = zeros(1,4);
% GMModels = cell(1,4);
% options = statset('MaxIter',500);
% for k = 1:4
%     GMModels{k} = fitgmdist(X,k,'Options',options,'CovarianceType','diagonal');
%     AIC(k)= GMModels{k}.AIC;
% end
% 
% [minAIC,numComponents] = min(AIC);
% numComponents
% 
% BestModel = GMModels{numComponents}
% evalclusters

%% 
%% plot features count and observe
mean_words_female = mean(words_train(logical(genders_train),:));
mean_words_male = mean(words_train(~logical(genders_train),:));

mean_image_features_f = mean(image_features_train(logical(genders_train),:));
mean_image_features_m = mean(image_features_train(~logical(genders_train),:));

var_image_features_f = var(image_features_train(logical(genders_train),:));
var_image_features_m = var(image_features_train(~logical(genders_train),:));
%%
figure;
plot(1:7, var_image_features_f,'bo');
hold on
plot(1:7, var_image_features_m,'rx');
hold off

figure;
plot(1:7, mean_image_features_f,'bo');
hold on
plot(1:7, mean_image_features_m,'rx');

%% 
mean_words_diff = abs(mean_words_female - mean_words_male);
figure;
plot(1:5001, mean_words_diff);
[V, I] = sort(mean_words_diff,'descend' );
X_words = words_train(I

%%

[n, ~] = size(words_train);
[parts] = make_xval_partition(n, 8);
clc
acc_ens=zeros(8,1);


%bns = calc_bns(words_train,Y);
IG=calc_information_gain(genders_train,words_train,[1:5000],10);
[top_bans, idx]=sort(IG,'descend');
%words_train_s=bsxfun(@times,words_train,IG);
acc = zeros(8,20);
for i=1:8
    row_sel1=(parts~=i);
    row_sel2=(parts==i);
    cols_sel=idx(1:500);
    
    Xtrain=words_train(row_sel1,cols_sel);
    Ytrain=genders_train(row_sel1);
    Xtest=words_train(row_sel2,cols_sel);
    Ytest=genders_train(row_sel2);
    
    % test 20 clusters
    for j = 1:20
        Yhat = gaussian_mixture(Xtrain,Ytrain,Xtest,Ytest, j*5);
        acc(i,j)=sum(round(Yhat)==Ytest)/length(Ytest);
    end
    %confusionmat(Ytest,Yhat)
end
acc;
mean(acc);


%% test Gaussian Mixture 
close all

X_s = X_selected;
%X_s = X_train_imagefea_cont;
GMModel = fitgmdist(X_s,2,'Start', 'plus','RegularizationValue',0.1);
threshold = [0.4,0.6];
P = posterior(GMModel,X_s);

n = size(X_s,1);
[~,order]= sort(P(:,1));
%%

figure;
plot(1:n,P(order,1),'r-',1:n,P(order,2),'b-');
legend({'Cluster 1', 'Cluster 2'});
ylabel('Cluster Membership Score');
xlabel('Point Ranking');
title('GMM with Full Unshared Covariances');

%%
idx = cluster(GMModel,X_s);
idxBoth = find(P(:,1)>=threshold(1) & P(:,1)<=threshold(2));
numInBoth = numel(idxBoth)

figure;
gscatter(X_s(:,1),X_s(:,2),idx,'rb','+o',5);
hold on;
plot(X_s(idxBoth,1),X_s(idxBoth,2),'ko','MarkerSize',10);
legend({'Cluster 1','Cluster 2','Both Clusters'},'Location','SouthEast');
title('Scatter Plot - GMM with Full Unshared Covariances')
hold off;