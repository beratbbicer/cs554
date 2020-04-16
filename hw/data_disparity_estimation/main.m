% clear;
% clc;
% run('vlfeat/toolbox/vl_setup');
% cloth1 = imread('../data/cloth1.png');
% cloth2 = imread('../data/cloth2.png');
% cloth1_disp = imread('../data/cloth1_disp.png');
% cloth2_disp = imread('../data/cloth2_disp.png');
% cloth1_gray_sp = im2single(rgb2gray(cloth1));
% cloth2_gray_sp = im2single(rgb2gray(cloth2));
% [F1,D1] = vl_sift(cloth1_gray_sp);
% F1 = F1';
% D1 = D1';
% [F2,D2] = vl_sift(cloth2_gray_sp);
% F2 = F2';
% D2 = D2';
% sift_matching_threshold = 5;
% [matching_idx1, matching_idx2] = sift_matching(D1,D2,sift_matching_threshold);
% % Clean up matchings - remove duplicates
% matching_points1 = F1(matching_idx1, 1:2);
% matching_points2 = F2(matching_idx2, 1:2);
% [~, idx1] = unique(matching_points1, 'rows', 'first');
% [~, idx2] = unique(matching_points2, 'rows', 'first');
% if length(idx1) < length(idx2)
%     unique_idx = idx1;
% else
%     unique_idx = idx2;
% end
% matching_points1 = matching_points1(unique_idx,:);
% matching_points2 = matching_points2(unique_idx,:);
% clear F1 F2 D1 D2;

% epoch = 518;
% dist = 20;
% patch_size = 7;
% [h, ~, ~] = ransac_homography(matching_points1, matching_points2, 1, 0.1, epoch, 100, 1);
% [rectified_img, ~, ~] = rectification(h, cloth1);
% [x,y,~] = size(cloth1);
% rectified_img = imresize(rectified_img, [x y]);
% dm = correlation_matching(cloth2, rectified_img, patch_size, dist);
% I = mat2gray(dm);
% figure;
% imshow(I);
% title(['Patch size: ' string(patch_size) ', Threshold: ' string(dist) ', RANSAC Epochs: ' string(epoch)]);

% --------------------------------------------------------------------------------------------------------

% epoch = 731;
% dist = 25;
% patch_size = 7;
% [h, ~, ~] = ransac_homography(matching_points2, matching_points1, 1, 0.1, epoch, 100, 1);
% [rectified_img, ~, ~] = rectification(h, cloth2);
% [x,y,~] = size(cloth2);
% rectified_img = imresize(rectified_img, [x y]);
% dm = correlation_matching(cloth1, rectified_img, patch_size, dist);
% I = mat2gray(dm);
% figure;
% imshow(I);
% title(['Patch size: ' string(patch_size) ', Threshold: ' string(dist) ', RANSAC Epochs: ' string(epoch)]);

% #########################################################################################################

clear;
clc;
run('vlfeat/toolbox/vl_setup');
plastic1 = imread('../data/plastic1.png');
plastic2 = imread('../data/plastic2.png');
plastic1_disp = imread('../data/plastic1_disp.png');
plastic2_disp = imread('../data/plastic2_disp.png');
plastic1_gray_sp = im2single(rgb2gray(plastic1));
plastic2_gray_sp = im2single(rgb2gray(plastic2));
[F1,D1] = vl_sift(plastic1_gray_sp);
F1 = F1';
D1 = D1';
[F2,D2] = vl_sift(plastic2_gray_sp);
F2 = F2';
D2 = D2';
% sift_matching_threshold = 2;
% [matching_idx1, matching_idx2] = sift_matching(D1,D2,sift_matching_threshold);
% matching_points1 = F1(matching_idx1, 1:2);
% matching_points2 = F2(matching_idx2, 1:2);
% [~, idx1] = unique(matching_points1, 'rows', 'first');
% [~, idx2] = unique(matching_points2, 'rows', 'first');
% if length(idx1) < length(idx2)
%     unique_idx = idx1;
% else
%     unique_idx = idx2;
% end
% matching_points1 = matching_points1(unique_idx,:);
% matching_points2 = matching_points2(unique_idx,:);
% 
% epoch = 855;
% dist = 50;
% patch_size = 7;
% [h, ~, ~] = ransac_homography(matching_points1, matching_points2, 3, 0.25, epoch, 100, 1);
% [rectified_img, ~, ~] = rectification(h, plastic1);
% [x,y,~] = size(plastic1);
% rectified_img = imresize(rectified_img, [x y]);
% dm = correlation_matching(plastic2, rectified_img, patch_size, dist);
% I = mat2gray(dm);
% figure;
% imshow(I);
% title(['Patch size: ' string(patch_size) ', Threshold: ' string(dist) ', RANSAC Epochs: ' string(epoch)]);

% --------------------------------------------------------------------------------------------------------

sift_matching_threshold = 3;
[matching_idx1, matching_idx2] = sift_matching(D1,D2,sift_matching_threshold);
matching_points1 = F1(matching_idx1, 1:2);
matching_points2 = F2(matching_idx2, 1:2);
[~, idx1] = unique(matching_points1, 'rows', 'first');
[~, idx2] = unique(matching_points2, 'rows', 'first');
if length(idx1) < length(idx2)
    unique_idx = idx1;
else
    unique_idx = idx2;
end
matching_points1 = matching_points1(unique_idx,:);
matching_points2 = matching_points2(unique_idx,:);
clear F1 F2 D1 D2;

epoch = 782;
dist = 50;
patch_size = 7;
[h, ~, ~] = ransac_homography(matching_points2, matching_points1, 3, 0.25, epoch, 100, 1);
[rectified_img, ~, ~] = rectification(h, plastic2);
[x,y,~] = size(plastic2);
rectified_img = imresize(rectified_img, [x y]);
dm = correlation_matching(plastic1, rectified_img, patch_size, dist);
I = mat2gray(dm);
figure;
imshow(I);
title(['Patch size: ' string(patch_size) ', Threshold: ' string(dist) ', RANSAC Epochs: ' string(epoch)]);