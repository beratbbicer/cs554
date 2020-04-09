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
sift_matching_threshold = 5;
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

% sift_matching_threshold = 5; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 1, 0.1, 731, 100, 1); % best 1
% sift_matching_threshold = 5; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 1, 0.1, 739, 100, 1); % best 2
epochs = [731];
for j = 1:length(epochs)
    epoch = epochs(j);
    [h, inlier_idx, avg_inlier_error] = ransac_homography(matching_points1, matching_points2, 1, 0.1, epoch, 100, 1);
    [rectified_img, xmin, ymin] = rectification(h, cloth1);

    [x,~] = size(matching_points1);
    homography_matching1 = zeros(x,2);
    for i=1:x
        [xp, yp] = get_correspondance(h, matching_points1(i,1), matching_points1(i,2));
        homography_matching1(i,:) = [ceil(xp - xmin + 1), ceil(yp - ymin + 1)];
    end

    figure;
    [fLMedS,inliers] = estimateFundamentalMatrix(homography_matching1, matching_points2,'NumTrials',4000);
    subplot(121);
    imshow(rectified_img); 
    title('Inliers and Epipolar Lines in First Image'); hold on;
    plot(homography_matching1(inliers,1), homography_matching1(inliers,2),'go');

    epiLines = epipolarLine(fLMedS',matching_points2(inliers,:));
    points = lineToBorderPoints(epiLines,size(rectified_img));
    line(points(:,[1,3])',points(:,[2,4])');

    subplot(122); 
    imshow(cloth2);
    title('Inliers and Epipolar Lines in Second Image'); hold on;
    plot(matching_points2(inliers,1), matching_points2(inliers,2),'go');

    epiLines = epipolarLine(fLMedS,homography_matching1(inliers,:));
    points = lineToBorderPoints(epiLines,size(cloth2));
    line(points(:,[1,3])',points(:,[2,4])');
    truesize;
    sgtitle([string(epoch) 'Epochs']);
end