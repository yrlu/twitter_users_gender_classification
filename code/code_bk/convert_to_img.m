% Author: Max Lu
% Date: Nov 20

% img_r, img_g, img_b, and img_grey are all 100*100*size(train_x, 1);
% one could easily type the following to show img;
%        imshow(img_grey(:,:,i)) 
function [img_r, img_g, img_b, img_grey] = convert_to_img(train_x)
train_x_r = train_x(:,1:10000);
train_x_g = train_x(:,10001:20000);
train_x_b = train_x(:,20001:30000);

train_x_r_img = reshape(train_x_r,[size(train_x_r,1) 100 100]);
img_r = zeros(100,100, size(train_x_r,1));

train_x_g_img = reshape(train_x_g,[size(train_x_g,1) 100 100]);
img_g = zeros(100,100, size(train_x_g,1));

train_x_b_img = reshape(train_x_r,[size(train_x_b,1) 100 100]);
img_b = zeros(100,100, size(train_x_b,1));

img_grey = zeros(100,100, size(train_x_b,1));
for i=1:size(train_x_r_img,1)
%   i
  cur_row_r=train_x_r_img(i, :, :);
  cur_row_g=train_x_g_img(i, :, :);
  cur_row_b=train_x_b_img(i, :, :);
  img_r(:, :, i) = reshape(cur_row_r, [100, 100, 1]);
  img_g(:, :, i) = reshape(cur_row_g, [100, 100, 1]);
  img_b(:, :, i) = reshape(cur_row_b, [100, 100, 1]);
  img_grey(:,:,i) = (reshape(cur_row_r, [100, 100, 1])*0.2989+reshape(cur_row_g, [100, 100, 1])*0.5870+reshape(cur_row_b, [100, 100, 1])*0.1140);
end


img_r = img_r/255;
img_g = img_g/255;
img_b = img_b/255;
img_grey = img_grey/255;
end