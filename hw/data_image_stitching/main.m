run('vlfeat/toolbox/vl_setup');
close all
% image_sets = {'im1', 'im2'};
% data_loc = 'data/';
% ext = '.png';
% 
% for i = 1:size(image_sets, 2)
% orig_img = imread([data_loc image_sets{i} ext]);
% rgb_img = orig_img;
% figure; imshow(rgb_img);
% end

% TODO
% Stich code get rid of VL
% Alpha blending

IMG.im1_orig = imread('data/im1.png');
IMG.im2_orig = imread('data/im2.png');
IMG.im1 = single(rgb2gray(IMG.im1_orig));
IMG.im2 = single(rgb2gray(IMG.im2_orig));

im1 = im2single(IMG.im1_orig);
im2 = im2single(IMG.im2_orig);

[f1, d1] = vl_sift(IMG.im1);
[f2, d2] = vl_sift(IMG.im2);
f1 = f1'; f2 = f2'; d1 = d1'; d2 = d2';

% [matches, scores] = vl_ubcmatch(d1', d2');
% matches = matches(:, 1:50);
% mpts1 = f1(matches(1, :), 1:2);
% mpts2 = f2(matches(2, :), 1:2);
% show_match(mpts1, mpts2, IMG)

[matching_idx1, matching_idx2] = sift_matching(d1, d2, 2);
mpts1 = f1(matching_idx1, 1:2); mpts2 = f2(matching_idx2, 1:2);
[mpts1, mpts2] = clean_matches(mpts1, mpts2);

% subplot(2, 1, 1)
% show_match(mpts1, mpts2, IMG);
% title('SIFT Matches')

% 10, 0.25, 700, 700, 10
if isfile("Homo.mat")
    Homo = load("Homo.mat");
    Homo = Homo.Homo;
    H = Homo.H;
    in_idx = Homo.in_idx;
else
    [h, in_idx, avg_inlier_error] = ransac_homography(...
        mpts1, mpts2, 10, 0.25, 700, 700, 10);
    H = [h 1];
    H = reshape(H,3,3)';
    Homo.H = H;
    Homo.in_idx = in_idx;
    save('Homo.mat','Homo');
end

mpts1 = mpts1(in_idx, :);
mpts2 = mpts2(in_idx, :);

subplot(2, 1, 2)
show_match(mpts1, mpts2, IMG);
title('RANSAC Matches')

% size(inlier_idx)
% avg_inlier_error

% stitch(H, im1, im2)
mosaic = im1;
width = floor(size(im2, 2)); height = floor(size(im2, 1));
mosaic = padarray(mosaic, [0 width], 0, 'post');
mosaic = padarray(mosaic, [height 0], 0, 'both');
ones_row = ones(1, size(mosaic, 1));
for i = 1:size(mosaic, 2)
    is_row = repmat(i, [1 size(mosaic, 1)]);
    j_all = [is_row; (1:size(mosaic, 1)) - height; ones_row];
    p = H * j_all;
    p = floor(p ./ p(3, :));
    for j = 1:size(mosaic, 1)
        if p(1, j) > 0 && p(1, j) <= width && p(2, j) > 0 && p(2, j) <= height
            mosaic(j, i, :) = im2(p(2, j), p(1, j), :);
        end
    end
end
%crop
[row,col] = find(mosaic);
c = max(col(:));
d = max(row(:));
st=imcrop(mosaic, [1 1 c d]);
[row,col] = find(mosaic ~= 0);
a = min(col(:));
b = min(row(:));
st=imcrop(st, [a b size(st,1) size(st,2)]);
figure;
imshow(st);

% Instead of forward warping the original image pixels, the coordinates 
% for the new pixels are inverse-warped to coordinates that may lie in 
% between pixels in the original image. If so, the new pixel values are 
% obtained through linear interpolation.
% https://inst.eecs.berkeley.edu/~cs194-26/fa17/upload/files/proj6B/cs194-26-abw/
function stitch(H, im1, im2)
w = size(im2,2); h = size(im2,1);
box = [1 w h 1;
       1 1 w h;
       1 1 1 1];
box = inv(H) * box ;

% Divide by homogeneous z, homo --> cartesian
box(1,:) = box(1,:) ./ box(3,:) ;
box(2,:) = box(2,:) ./ box(3,:) ;

ur = min([1 box(1,:)]):max([size(im1,2) box(1,:)]) ;
vr = min([1 box(2,:)]):max([size(im1,1) box(2,:)]) ;

[u,v] = meshgrid(ur,vr) ;
im1_ = vl_imwbackward(im2double(im1),u,v) ;

z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
im2_ = vl_imwbackward(im2double(im2),u_,v_) ;

% Calculate combined image pixel values for  blending
alpha = 0.5;
mass = alpha * ~isnan(im1_) + (1 - alpha) * ~isnan(im2_);
mass = ~isnan(im1_) + ~isnan(im2_);

% Blackout NANs 
im1_(isnan(im1_)) = 0 ;
im2_(isnan(im2_)) = 0 ;

% Combine images
% mosaic = (im1_ + im2_);

% Alpha Blending
mosaic = (im1_ + im2_) ./ 2.*mass ;

figure(2) ; clf ;
imagesc(mosaic) ; axis image off ;
title('Mosaic') ;

end

function [mpts1, mpts2] = clean_matches(mpts1, mpts2)
[~, idx1] = unique(mpts1, 'rows', 'first');
[~, idx2] = unique(mpts2, 'rows', 'first');
if length(idx1) < length(idx2)
    unique_idx = idx1;
else
    unique_idx = idx2;
end
mpts1 = mpts1(unique_idx,:);
mpts2 = mpts2(unique_idx,:);
end

function f = show_match(mpts1, mpts2, IMG)
adjusted_im1 = [IMG.im1_orig; zeros(1, size(IMG.im1_orig, 2), ...
    size(IMG.im1_orig, 3))];
numPoints = size(mpts1, 1);
offset_width = size(IMG.im1, 2);
% figure;
imshow([adjusted_im1 IMG.im2_orig]);
hold on;
for i = 1 : numPoints
    plot(mpts1(i, 1), mpts1(i, 2), 'b*', mpts2(i, 1) + offset_width, ...
         mpts2(i, 2), 'r*');
    line([mpts1(i, 1) mpts2(i, 1) + offset_width], ...
        [mpts1(i, 2) mpts2(i, 2)], ...
         'Color', 'green');
end
hold off;
end