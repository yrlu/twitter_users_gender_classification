% Author: Max Lu
% Date: Dec 5

function [hog_feat, certain, pca_hog] = gen_data_hog()
    load('certain_HOG.mat', 'eyes_hog', 'face_hog','nose_hog', 'certain');
    hog_feat = [face_hog nose_hog face_hog];
    hog_feat_certain = hog_feat(logical(certain),:);
    [U mu vars] = pca_1(hog_feat_certain');
    [pca_hog,Xhat,avsq] = pcaApply(hog_feat', U, mu, 1500);
    pca_hog = double(pca_hog');
end
