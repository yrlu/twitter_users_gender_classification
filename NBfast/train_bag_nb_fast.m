%% NB bagging fast, assumming the features have been selected
% Deng, Xiang
function [ models ] = train_bag_nb_fast( Xtrainset,Ytrain,F,M )
Nx=size(Xtrainset,1); % Total n


for m=1:M
    
    ind = randperm(Nx);
    subset_size = floor(Nx * F);
    ind_clip = ind(1:subset_size);
    
    Y_cur = Ytrain(ind_clip);
    X_cur = Xtrainset(ind_clip,:);
    
    model = train_fastnb(sparse(X_cur), Y_cur, [0 1]);

    models.NB{m}=model; 
end
end

