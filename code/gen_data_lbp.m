% Author: Max Lu
% Date: Dec 5

function [lbp_feat, pca_lbp] = gen_data_lbp()
load('img_pca_basis_lbp.mat', 'U_lbp', 'mu_lbp', 'vars_lbp');
load('lbp_feat.mat','lbp_feat');

[pca_lbp,Xhat,avsq] = pcaApply(lbp_feat', U_lbp, mu_lbp, 2000);
pca_lbp = double(pca_lbp');

end