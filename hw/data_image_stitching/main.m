run('vlfeat/toolbox/vl_setup');
% close all
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

subplot(2, 1, 1)
show_match(mpts1, mpts2, IMG);
title('SIFT Matches')

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

stitch(H, im1, im2)

% % alpha = 1;
% alpha = 0;
% W = abs(colmin2 - colmax1);
% W = floor((1-alpha)*W);
% alpha_mask1 = flip(1:W)./W;
% z2 = zeros(1, size(im2mask, 2) - colmax1, 1);
% z1 = ones(1, size(im2mask, 2) - size(z2, 2) - size(alpha_mask1, 2));
% alpha_mask1 = [z1 alpha_mask1 z2];
% alpha_mask1 = repmat(alpha_mask1, [size(im2mask, 1) 1]);
% alpha_mask1 = im1mask .* alpha_mask1;
% 
% % W = abs(colmax1 - colmin2);
% W = floor((alpha)*W);
% alpha_mask2 = (1:W)./W;
% z2 = ones(1, size(im2mask, 2) - colmax1, 1);
% z1 = zeros(1, size(im2mask, 2) - size(z2, 2) - size(alpha_mask2, 2));
% alpha_mask2 = [z1 alpha_mask2 z2];
% alpha_mask2 = repmat(alpha_mask2, [size(im2mask, 1) 1]);
% alpha_mask2 = im2mask .* alpha_mask2;

%       if im1mask(j, i) == 1 && im2mask(j, i) == 1
%           alpha_mask1(j, i) = alpha_mask1(j, i);
%           alpha_mask2(j, i) = alpha_mask2(j, i);
%       end

function stitch(H, im1, im2)
% Extend the image so that the final result can accomadate the affine
% transformation of im2. Obtain 'mosaic'
width = floor(size(im2, 2)); height = floor(size(im2, 1));

mosaic = pad_affine(im1, width, height);

% Create a mask for mosaic whose elements are '1' if im1 has a pixel there
im1mask = pad_affine(ones(size(im1)), width, height);
im1mask = im1mask(:, :, 1);
% Create a masks for mosaic, will be populated while doing trans. to im2
im2mask = zeros(size(im1mask));

ow_mos = mosaic;
shi_mos = mosaic;
alpha_mos = mosaic;
just_im2 = zeros(size(mosaic));
ones_row = ones(1, size(mosaic, 1));
for i = 1:size(mosaic, 2)
    is_row = repmat(i, [1 size(mosaic, 1)]);
    j_all = [is_row; (1:size(mosaic, 1)) - height; ones_row];
    p = H * j_all;
    p = floor(p ./ p(3, :));
    for j = 1:size(mosaic, 1)
        if p(1, j) > 0 && p(1, j) <= width && p(2, j) > 0 && p(2, j) <= height
            im2_pixel = im2(p(2, j), p(1, j), :);
            just_im2(j, i, :) = im2_pixel;
            ow_mos(j, i, :) = im2_pixel;
            shi_mos(j, i, :) = mosaic(j, i, :) + im2_pixel;
            im2mask(j, i) = 1;
        end
    end
end

[~,col] = find(im1mask ~= 0);
colmin1 = min(col(:)); colmax1 = max(col(:));
[~,col] = find(im2mask ~= 0);
colmin2 = min(col(:)); colmax2 = max(col(:));

alpha = 0.7;
W = abs(colmin2 - colmax1);
alpha_mask1 = flip(1:W)./W;
z1 = ones(1, colmin2);
z2 = zeros(1, size(im2mask, 2) - colmax1, 1);
alpha_mask1 = [z1 alpha_mask1 z2];
alpha_mask1 = repmat(alpha_mask1, [size(im2mask, 1) 1]);
alpha_mask1 = im1mask .* alpha_mask1;

W = abs(colmax1 - colmin2);
alpha_mask2 = (1:W)./W;
z1 = zeros(1, colmin2);
z2 = ones(1, size(im2mask, 2) - colmax1, 1);
alpha_mask2 = [z1 alpha_mask2 z2];
alpha_mask2 = repmat(alpha_mask2, [size(im2mask, 1) 1]);
alpha_mask2 = im2mask .* alpha_mask2;

for i = 1:size(mosaic, 2)
   for j = 1:size(mosaic, 1)
      if im1mask(j, i) == 1 && im2mask(j, i) == 0
          alpha_mask1(j, i) = 1;
      end
      if im1mask(j, i) == 0 && im2mask(j, i) == 1
          alpha_mask2(j, i) = 1;
      end
   end
end

blend_im1 = mosaic .* alpha_mask1;
blend_im2 = just_im2 .* alpha_mask2;
mosaic = (blend_im1 + blend_im2);

% 0--> empty 1--> pixel on an im 2--> pixel on both im
norm_mask = im1mask + im2mask;
norm_mask(norm_mask == 2) = 0.5;
norm_mask = norm_mask(:, :, 1);

% mosaic = shi_mos .* norm_mask;
% figure; imshow(crop_blacks(ow_mos));
% figure; imshow(crop_blacks(shi_mos));
% figure; imshow(crop_blacks(mosaic));
% figure; imshow(crop_blacks(alpha_mos));
figure; imshow(crop_blacks(mosaic));
end

function M = pad_affine(M, W, H)
M = padarray(M, [0 W], 0, 'post');
M = padarray(M, [H 0], 0, 'both');
end

function mosaic = crop_blacks(mosaic)
[row,col] = find(mosaic ~= 0);
a = min(col(:)); b = min(row(:));
c = max(col(:)); d = max(row(:));
mosaic = imcrop(mosaic, [a b c-a d-b]);
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