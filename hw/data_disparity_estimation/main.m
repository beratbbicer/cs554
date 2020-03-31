% vl_plotsiftdescriptor(D,F);
clear;
clc;
run('vlfeat/toolbox/vl_setup');
cloth1 = imread('../data/cloth1.png');
cloth2 = imread('../data/cloth2.png');
cloth1_disp = imread('../data/cloth1_disp.png');
cloth2_disp = imread('../data/cloth2_disp.png');
cloth1_gray_sp = im2single(rgb2gray(cloth1));
cloth2_gray_sp = im2single(rgb2gray(cloth2));
[F1,D1] = vl_sift(cloth1_gray_sp);
F1 = F1';
D1 = D1';
[F2,D2] = vl_sift(cloth2_gray_sp);
F2 = F2';
D2 = D2';
sift_matching_threshold = 7;
[matching_idx1, matching_idx2] = sift_matching(D1,D2,sift_matching_threshold);
matching_points1 = F1(matching_idx1, 1:2);
[~,IA,~] = unique(matching_points1, 'rows');
matching_points1 = matching_points1(IA, :);
matching_points2 = F2(matching_idx2, 1:2);
matching_points2 = matching_points2(IA, :);
clear F1 F2 D1 D2 IA;
[h, idx] = ransac_homography(matching_points1, matching_points2, 80, 0.4, 1500, 50, 0.5);
for i = 1:length(idx)
    [x_p, y_p] = get_correspondance(h, matching_points1(idx(i),1), matching_points1(idx(i),2));
    disp([matching_points2(idx(i),1) x_p matching_points2(idx(i),2) y_p]);
end