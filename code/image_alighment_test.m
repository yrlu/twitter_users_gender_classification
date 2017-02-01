%%
%% 
tic
train_x = [images_train; images_train(1,:); images_train(2,:)];
train_y = [genders_train; genders_train(1); genders_train(2)];
test_x =  images_test;

[train_r, train_g, train_b, train_grey] = convert_to_img(train_x);
[test_r, test_g, test_b, test_grey] = convert_to_img(test_x);
toc
%%
tic

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the detector.
% videoFileReader = vision.VideoFileReader('visionface.avi');
% videoFrame      = step(videoFileReader);
certain = ones(size(train_grey,3)+size(test_grey,3),1);
for i  = 1:size(train_grey,3)
profile = train_grey(:,:, i);
bbox            = step(faceDetector, profile);

if ~isempty(bbox)
    i
% Draw the returned bounding box around the detected face.
% videoOut = insertObjectAnnotation(profile,'rectangle',bbox,'Face');
size(bbox);
bbox;
profile = imcrop(profile,bbox(1,:));
profile=imresize(profile,[100 100]);
train_grey(:,:, i) = profile;
else
    bbox_r            = step(faceDetector, train_r(:,:, i));
    bbox_g            = step(faceDetector, train_g(:,:, i));
    bbox_b            = step(faceDetector, train_b(:,:, i));
    if ~isempty(bbox_r)
        profile = imcrop(profile,bbox_r(1,:));
        profile=imresize(profile,[100 100]);
        train_grey(:,:, i) = profile;
    elseif ~isempty(bbox_g)
        profile = imcrop(profile,bbox_g(1,:));
        profile=imresize(profile,[100 100]);
        train_grey(:,:, i) = profile;
    elseif ~isempty(bbox_b)
        profile = imcrop(profile,bbox_b(1,:));
        profile=imresize(profile,[100 100]);
        train_grey(:,:, i) = profile;
    else
        certain(i) = 0;
    end
end
% imshow(profile), title('Detected face');
end 
 %save('train_grey_faces.mat', 'train_grey');
 save('train_grey_certain.mat', 'certain');
toc

%%
load('train_grey_faces.mat', 'train_grey');

certain_train = certain(1:size(train_grey,3));

%%
train_grey_certain = train_grey(:,:,logical(certain_train));
train_y_certain = train_y(logical(certain_train));

save certain.mat train_grey_certain train_y_certain 
%%
for i = 1:size(train_grey_certain,3)
    i
   imshow(train_grey_certain(:,:,i)); 
end

%% What I need is
load certain.mat train_grey_certain train_y_certain 
% train_grey_certain = train_grey(:,:,logical(certain_train));
% train_y_certain = train_y(logical(certain_train));

[h, w, n] = size(train_grey_certain);
image_train_X = reshape(train_grey_certain,[h*w n]);
image_train_Y = train_y_certain;%[genders_train;genders_train(1);genders_train(2)];

size(image_train_X)
size(image_train_Y)

%%
figure;
imshow(train_grey_certain(:,:,1)); 
figure;
imshow(train_grey_certain(:,:,4)); 

%% 
fixed = train_grey_certain(:,:,1);
moving = train_grey_certain(:,:,4);
imshowpair(fixed, moving,'Scaling','joint');
[optimizer, metric] = imregconfig('multimodal');
movingRegistered = imregister(moving, fixed, 'affine', optimizer, metric);
%%
figure;
imshowpair(fixed, movingRegistered,'Scaling','joint');
%%
movingTo = zeros(100,100,3640);
for i = 2:3640
    tic
    moving = train_grey_certain(:,:,i);
    movingTo(:,:,i) = imregister(moving, fixed, 'affine', optimizer, metric);
    toc
end

%%
figure;
imshow(train_grey_certain(:,:,7));
%imshow(movingTo(:,:,7));

imshow(edge(train_grey_certain(:,:,7),'Canny'));