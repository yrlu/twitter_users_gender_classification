for i=1:size(img_X,1)
  cur_row=img_X(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
  imshow(uint8(cur_img));
  pause(3);
end
