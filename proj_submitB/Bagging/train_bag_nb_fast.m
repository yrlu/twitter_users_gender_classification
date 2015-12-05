%% NB bagging fast
% Deng, Xiang
function [ models,col_sel,bns_selected ] = train_bag_nb_fast( Xtrainset,Ytrain,num_bns,F,M )
Nx=size(Xtrainset,1); % Total n

bns = calc_bns(Xtrainset,Ytrain,0.05);
bns=bns/max(bns);
[top_bns, idx]=sort(bns,'descend');
col_sel=idx(1:num_bns);
bns_selected=bns(col_sel);
for m=1:M
    
    ind = randperm(Nx);
    subset_size = floor(Nx * F);
    ind_clip = ind(1:subset_size);
    
    Y_cur = Ytrain(ind_clip);
    X_cur = Xtrainset(ind_clip,col_sel);
    X_cur=bsxfun(@times,X_cur,bns(col_sel) );%------scale the columns by bns_i s
    X_cur=round(X_cur);
    model = train_fastnb(sparse(X_cur), Y_cur, [0 1]);
    
    models.NB{m}=model;
end
end

