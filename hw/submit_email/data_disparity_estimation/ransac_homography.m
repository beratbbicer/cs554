function [h, inlier_idx, inlier_error] = ransac_homography(pts1, pts2, threshold, minpts_ratio, h_iternum, iternum, lambda)
% New approach: Fit homography once!
rng(1);
[x, ~] = size(pts1);
minpts = round(x * minpts_ratio);

if minpts < 8
    minpts = min(8, x);
end

error = zeros(iternum,x);
models = zeros(iternum, 8);
avg_error = ones(iternum,1) * 10000;

for it = 1:iternum
    initin_idx = randperm(x,minpts);
    h = fit_homography(pts1(initin_idx, :), pts2(initin_idx,:), h_iternum, lambda);
    
    for i = 1:x
        [x_p,y_p] = get_correspondance(h,pts1(i,1),pts1(i,2));
        error(it, i) = distance_homography(pts2(i,1), pts2(i,2), x_p, y_p);
    end
    
    in_idx = find(error(it,:) < threshold);
    
    if length(in_idx) < minpts
        continue
    end
    
    avg_error(it) = mean(error(it,:));
    models(it,:) = h; % fit_homography(pts1(in_idx, :), pts2(in_idx,:), h_iternum, lambda);
end

[~,idx] = min(avg_error);
h = models(idx,:);
inlier_idx = find(error(idx,:) < threshold);
inlier_error = mean(error(idx, inlier_idx));
end

