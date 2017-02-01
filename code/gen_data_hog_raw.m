function [hog_feat, certain] = gen_data_hog_raw()
load('certain_HOG.mat', 'eyes_hog', 'face_hog','nose_hog', 'certain', 'U', 'mu');
hog_feat = [face_hog eyes_hog nose_hog];
end