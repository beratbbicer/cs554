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
sift_matching_threshold = 2;
[matching_idx1, matching_idx2] = sift_matching(D1,D2,sift_matching_threshold);
% Clean up matchings - remove duplicates
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

% 2 -> 1
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 731, 100, 1); % good
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 790, 100, 1);
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 855, 100, 1);
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 884, 100, 1);
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 941, 100, 1);
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 953, 100, 1);

% 1 -> 2
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 843, 100, 1);
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 855, 100, 1);
% sift_matching_threshold = 2; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 3, 0.25, 860, 100, 1);

epoch = 843;
dist = 30;
patch_size = 9;
[h, ~, avg_inlier_error] = ransac_homography(matching_points2, matching_points1, 3, 0.25, epoch, 100, 1);
[rectified_img, ~, ~] = rectification(h, plastic2);
[x,y,~] = size(plastic2);
rectified_img = imresize(rectified_img, [x y]);
matching_pts = correlation_matching(plastic1, rectified_img, avg_inlier_error, patch_size, dist);
dm = disparity_map(matching_pts, plastic1, 1000);
I = mat2gray(dm);
figure;
imshow(I);
title(['Patch size: ' string(patch_size) ', Threshold: ' string(dist) ', RANSAC Epochs: ' string(epoch)]);

% -------------------------------------

epoch = 855;
dist = 20;
patch_size = 15;
[h, ~, avg_inlier_error] = ransac_homography(matching_points2, matching_points1, 3, 0.25, epoch, 100, 1);
[rectified_img, ~, ~] = rectification(h, plastic2);
[x,y,~] = size(plastic2);
rectified_img = imresize(rectified_img, [x y]);
matching_pts = correlation_matching(plastic1, rectified_img, avg_inlier_error, patch_size, dist);
dm = disparity_map(matching_pts, plastic1, 1000);
I = mat2gray(dm);
figure;
imshow(I);
title(['Patch size: ' string(patch_size) ', Threshold: ' string(dist) ', RANSAC Epochs: ' string(epoch)]);

% -------------------------------------

epoch = 860;
dist = 20;
patch_size = 15;
[h, ~, avg_inlier_error] = ransac_homography(matching_points2, matching_points1, 3, 0.25, epoch, 100, 1);
[rectified_img, ~, ~] = rectification(h, plastic2);
[x,y,~] = size(plastic2);
rectified_img = imresize(rectified_img, [x y]);
matching_pts = correlation_matching(plastic1, rectified_img, avg_inlier_error, patch_size, dist);
dm = disparity_map(matching_pts, plastic1, 1000);
I = mat2gray(dm);
figure;
imshow(I);
title(['Patch size: ' string(patch_size) ', Threshold: ' string(dist) ', RANSAC Epochs: ' string(epoch)]);

%     figure;
%     [fLMedS,inliers] = estimateFundamentalMatrix(homography_matching2, matching_points1,'NumTrials',4000);
%     subplot(121);
%     imshow(rectified_img); 
%     title('Inliers and Epipolar Lines in First Image'); hold on;
%     plot(homography_matching2(inliers,1), homography_matching2(inliers,2),'go');
% 
%     epiLines = epipolarLine(fLMedS',matching_points1(inliers,:));
%     points = lineToBorderPoints(epiLines,size(rectified_img));
%     line(points(:,[1,3])',points(:,[2,4])');
% 
%     subplot(122); 
%     imshow(plastic1);
%     title('Inliers and Epipolar Lines in Second Image'); hold on;
%     plot(matching_points1(inliers,1), matching_points1(inliers,2),'go');
% 
%     epiLines = epipolarLine(fLMedS,homography_matching2(inliers,:));
%     points = lineToBorderPoints(epiLines,size(plastic1));
%     line(points(:,[1,3])',points(:,[2,4])');
%     truesize;
%     sgtitle([string(epoch) 'Epochs']);

% tmp = [50];
% patch_sizes = [9];
% for i = 1:length(tmp)
%     dist = tmp(i);
%     figure;
%     for j = 1:length(patch_sizes)
%         patch_size = patch_sizes(j);
%         [x,y,z] = size(plastic1);
%         rectified_img = imresize(rectified_img, [x y]);
%         matching_pts = correlation_matching(plastic1, rectified_img, avg_inlier_error, patch_size, dist);
%         dm = disparity_map(matching_pts, plastic1, 1000);
%         I = mat2gray(dm);
%         subplot(1,3,j);
%         imshow(I);
%         title(['Patch size ' string(patch_size) ', Threshold ' string(dist)]);
%     end
%     sgtitle(['Distance threshold = ' string(dist)]);
% end
