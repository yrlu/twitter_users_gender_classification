%% Ensemble trees
% Deng, Xiang
function [ models, cols_sel ] = train_bag_tree( Xtrainset,Ytrain,numTree,num_ig,F,M )
Nx=size(Xtrainset,1); % Total n
% Feature selection
bns = calc_bns(Xtrainset,Ytrain);
IG=calc_information_gain(Ytrain,Xtrainset,[1:size(Xtrainset,2)],10);
[top_igs, idx]=sort(IG,'descend');
cols_sel=idx(1:num_ig);

for m=1:M
    
    ind = randperm(Nx);
    subset_size = floor(Nx * F);
    ind_clip = ind(1:subset_size);
    
    Y_cur = Ytrain(ind_clip);
    X_cur = Xtrainset(ind_clip,:);
    
    % Feature selection
%     bns = calc_bns(X_cur,Y_cur);
%     IG=calc_information_gain(Y_cur,X_cur,[1:size(X_cur,2)],10);
%     [top_igs, idx]=sort(IG,'descend');
%     cols_sel=idx(1:num_ig);
%     
    X_cur = X_cur(:,cols_sel);    
    
    ens = fitensemble(X_cur,Y_cur,'LogitBoost',numTree,'Tree' );
    models.enstree{m}=ens; 
    models.cols_sel=cols_sel;
end
end

