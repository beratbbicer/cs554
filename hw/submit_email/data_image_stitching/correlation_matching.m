function matching_pts = correlation_matching(img1, img2, avg_inlier_error, template_size)
% Obtain a rectangular, odd-sized search window for matches
[x1,y1] = size(img1);
[x2, y2] = size(img2);
xdiff = ceil(abs(x1 - x2));
ydiff = ceil(abs(y1 - y2));
inlier_error_window = ceil(avg_inlier_error);
search_window =  [xdiff + inlier_error_window ydiff + inlier_error_window];
if search_window(1) < template_size search_window(1) = template_size; end
if search_window(2) < template_size search_window(2) = template_size; end
if mod(search_window(1),2) == 0 search_window(1) = search_window(1) + 1; end
if mod(search_window(2),2) == 0 search_window(2) = search_window(2) + 1; end
end

