run('vlfeat/toolbox/vl_setup');

% image_sets = {'im1', 'im2'};
% data_loc = 'data/';
% ext = '.png';
% 
% for i = 1:size(image_sets, 2)
% orig_img = imread([data_loc image_sets{i} ext]);
% rgb_img = orig_img;
% figure; imshow(rgb_img);
% end

im1_orig = imread('data/im1.png');
im2_orig = imread('data/im2.png');
im1 = single(rgb2gray(im1_orig));
im2 = single(rgb2gray(im2_orig));
adjusted_im1 = [im1_orig; zeros(1, size(im1_orig, 2), size(im1_orig, 3))];

[f1, d1] = vl_sift(im1);
[f2, d2] = vl_sift(im2);
f1 = f1'; f2 = f2'; d1 = d1'; d2 = d2';
[matches, scores] = vl_ubcmatch(d1', d2');
[matching_idx1, matching_idx2] = sift_matching(d1, d2, 1);
mpts1 = f1(matching_idx1, 1:2);
mpts2 = f2(matching_idx2, 1:2);
[~, idx1] = unique(mpts1, 'rows', 'first');
[~, idx2] = unique(mpts2, 'rows', 'first');
if length(idx1) < length(idx2)
    unique_idx = idx1;
else
    unique_idx = idx2;
end
mpts1 = mpts1(unique_idx,:);
mpts2 = mpts2(unique_idx,:);

mpts1 = mpts1(1:50, :);
mpts2 = mpts2(1:50, :);

numPoints = size(mpts1, 1);
offset_width = size(im1, 2);
figure; imshow([adjusted_im1 im2_orig]);
hold on;
for i = 1 : numPoints
    plot(mpts1(i, 1), mpts1(i, 2), 'b*', mpts2(i, 1) + offset_width, ...
         mpts2(i, 2), 'r*');
    line([mpts1(i, 1) mpts2(i, 1) + offset_width], ...
        [mpts1(i, 2) mpts2(i, 2)], ...
         'Color', 'green');
end
hold off;

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
% 
% % sift_matching_threshold = 5; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 1, 0.1, 731, 100, 1); % best 1
% % sift_matching_threshold = 5; [h, inlier_idx] = ransac_homography(matching_points1, matching_points2, 1, 0.1, 739, 100, 1); % best 2
% epochs = [731];
% for j = 1:length(epochs)
%     epoch = epochs(j);
%     [h, inlier_idx, avg_inlier_error] = ransac_homography(matching_points1, matching_points2, 1, 0.1, epoch, 100, 1);
%     [rectified_img, xmin, ymin] = rectification(h, cloth1);
% 
%     [x,~] = size(matching_points1);
%     homography_matching1 = zeros(x,2);
%     for i=1:x
%         [xp, yp] = get_correspondance(h, matching_points1(i,1), matching_points1(i,2));
%         homography_matching1(i,:) = [ceil(xp - xmin + 1), ceil(yp - ymin + 1)];
%     end
% 
%     figure;
%     [fLMedS,inliers] = estimateFundamentalMatrix(homography_matching1, matching_points2,'NumTrials',4000);
%     subplot(121);
%     imshow(rectified_img); 
%     title('Inliers and Epipolar Lines in First Image'); hold on;
%     plot(homography_matching1(inliers,1), homography_matching1(inliers,2),'go');
% 
%     epiLines = epipolarLine(fLMedS',matching_points2(inliers,:));
%     points = lineToBorderPoints(epiLines,size(rectified_img));
%     line(points(:,[1,3])',points(:,[2,4])');
% 
%     subplot(122); 
%     imshow(cloth2);
%     title('Inliers and Epipolar Lines in Second Image'); hold on;
%     plot(matching_points2(inliers,1), matching_points2(inliers,2),'go');
% 
%     epiLines = epipolarLine(fLMedS,homography_matching1(inliers,:));
%     points = lineToBorderPoints(epiLines,size(cloth2));
%     line(points(:,[1,3])',points(:,[2,4])');
%     truesize;
%     sgtitle([string(epoch) 'Epochs']);
% end