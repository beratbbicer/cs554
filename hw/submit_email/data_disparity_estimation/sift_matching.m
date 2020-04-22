function [matching_idx1, matching_idx2] = sift_matching(pts1, pts2,threshold)
[size1,~] = size(pts1);
[size2,~] = size(pts2);
matching_idx1 = [];
matching_idx2 = [];

for i = 1:size1
    min_dist = inf; prev_min_dist = inf; min_index = 1;
    for j = 1:size2
        distance = norm(double(pts1(i,:) - pts2(j,:)),2);
        if distance < min_dist
            prev_min_dist = min_dist;
            min_dist = distance;
            min_index = j;
        end
    end
    
    if prev_min_dist / min_dist > threshold
        matching_idx1 = [matching_idx1 i];
        matching_idx2 = [matching_idx2 min_index];
    end
end
end