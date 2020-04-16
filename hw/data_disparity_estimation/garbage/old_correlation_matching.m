function disparity_map = correlation_matching(img1, img2, avg_inlier_error, template_size, y_dist_threshold)
% Obtain a rectangular, odd-sized search window for matches
[x1,y1,~] = size(img1);
[x2,y2,~] = size(img2);
xdiff = ceil(abs(x1 - x2));
ydiff = ceil(abs(y1 - y2));
inlier_error_window = ceil(avg_inlier_error);
search_window =  [xdiff + inlier_error_window ydiff + inlier_error_window];
if search_window(1) < template_size, search_window(1) = template_size + 1; end
if search_window(2) < template_size, search_window(2) = template_size + 1; end
if mod(search_window(1),2) == 0, search_window(1) = search_window(1) + 1; end
if mod(search_window(2),2) == 0, search_window(2) = search_window(2) + 1; end
tmp_width = floor(template_size / 2);
disparity_map = zeros(x1,y1);
img1 = im2double(img1);
img2 = im2double(img2);
for i=1:x1
    for j=1:y1
       % Construct and normalize template over channels
       template_x = i-tmp_width : 1 : i+tmp_width;
       template_y = j-tmp_width : 1 : j+tmp_width;
       mask = ones(2*tmp_width+1);   
       
       if i-tmp_width < 1
           template_x = 1 : 1 : i+tmp_width;
           mask(1:1:1-(i-tmp_width),:) = 0; 
       end
       
       if i+tmp_width > x1
           template_x = i-tmp_width : 1: x1;
           
           midpoint = ceil(template_size/2);
           del_rows_count = (i + tmp_width) - x1;
           del_rows = [];
           q = 0;
           
           while q < del_rows_count
               del_rows = [del_rows; template_size - q];
               q = q + 1;
           end
           
           mask(del_rows, :) = 0;
       end
       
       if j-tmp_width < 1 
           template_y = 1 : 1 : j+tmp_width;
           mask(:, 1:1:1-(j-tmp_width),:) = 0;
       end
       
       if j+tmp_width > y1
           template_y = j-tmp_width:1:y1;
           midpoint = ceil(template_size/2);
           del_cols_count = (j + tmp_width) - y1;
           del_cols = [];
           q = 0;
           
           while q < del_cols_count
               del_cols = [del_cols; template_size - q];
               q = q + 1;
           end
           
           mask(:, del_cols) = 0;
       end
       
       tmp = img1(template_x, template_y, :);
       rtmp = tmp(:,:,1);
       normx_rtmp = rtmp - min(rtmp(:));
       tmp(:,:,1) = normx_rtmp; % normx_rtmp ./ (max(normx_rtmp(:)) + 0.0001);
       gtmp = tmp(:,:,2);
       normx_gtmp = gtmp - min(gtmp(:));
       tmp(:,:,2) = normx_gtmp; % normx_gtmp ./ (max(normx_gtmp(:)) + 0.0001);
       btmp = tmp(:,:,3);
       normx_btmp = btmp - min(btmp(:));
       tmp(:,:,3) = normx_btmp; % normx_btmp ./ (max(normx_btmp(:)) + 0.0001);
       clear rtmp normx_rtmp gtmp normx_gtmp btmp normx_btmp;
       
       search_y = j - y_dist_threshold:1:j+y_dist_threshold;
       if j - y_dist_threshold < 1, search_y = 1:1:j+y_dist_threshold; end
       if j + y_dist_threshold > y2, search_y = j - y_dist_threshold:1:y2; end
          
       % Obtain filter responses
       responses = ones(y2,1);
       for jj = 1:length(search_y)
           x_img2 = i;
           y_img2 = search_y(jj);
           
           if img2(x_img2, y_img2, 1) == 0, continue; end
               
           patch_mask = ones(template_size);
           if x_img2-tmp_width < 1
               patch_mask(1:1:1-(x_img2-tmp_width),:) = 0; 
           end

           if x_img2+tmp_width > x2
               midpoint = ceil(template_size/2);
               del_rows_count = (x_img2+tmp_width) - x2;
               del_rows = [];
               q = 0;

               while q < del_rows_count
                   del_rows = [del_rows; template_size - q];
                   q = q + 1;
               end

               patch_mask(del_rows, :) = 0;
           end

           if y_img2-tmp_width < 1
               patch_mask(:, 1:1:1-(y_img2-tmp_width)) = 0;
           end

           if y_img2+tmp_width > y2
               midpoint = ceil(template_size/2);
               del_cols_count = (y_img2 + tmp_width) - y2;
               del_cols = [];
               q = 0;

               while q < del_cols_count
                   del_cols = [del_cols; template_size - q];
                   q = q + 1;
               end

               patch_mask(:, del_cols) = 0;
           end

           response_mask = mask .* patch_mask;
           [rows, cols] = ind2sub([template_size template_size],find(response_mask ~= 0));
           rows_idx = rows - ceil(template_size/2);
           cols_idx = cols - ceil(template_size/2);
           patch_response_r = 0;
           patch_response_g = 0;
           patch_response_b = 0;
           for q = 1:length(rows)
%                patch_response_r = patch_response_r + (tmp(rows(q) - rows(1) + 1, cols(q) - cols(1) + 1, 1) - img2(x_img2 + rows_idx(q), y_img2 + cols_idx(q), 1))^2;
%                patch_response_g = patch_response_g + (tmp(rows(q) - rows(1) + 1, cols(q) - cols(1) + 1, 2) - img2(x_img2 + rows_idx(q), y_img2 + cols_idx(q), 2))^2;
%                patch_response_b = patch_response_b + (tmp(rows(q) - rows(1) + 1, cols(q) - cols(1) + 1, 3) - img2(x_img2 + rows_idx(q), y_img2 + cols_idx(q), 3))^2;
                 patch_response_r = patch_response_r + (tmp(rows(q) - rows(1) + 1, cols(q) - cols(1) + 1, 1) * img2(x_img2 + rows_idx(q), y_img2 + cols_idx(q), 1));
                 patch_response_g = patch_response_g + (tmp(rows(q) - rows(1) + 1, cols(q) - cols(1) + 1, 2) * img2(x_img2 + rows_idx(q), y_img2 + cols_idx(q), 2));
                 patch_response_b = patch_response_b + (tmp(rows(q) - rows(1) + 1, cols(q) - cols(1) + 1, 3) * img2(x_img2 + rows_idx(q), y_img2 + cols_idx(q), 3));
           end
           responses(jj) = mean([patch_response_r patch_response_g patch_response_b]);
       end
       
       % Get the patch center with hightest correspondance and closest to original point
       if max(responses(:)) == 0, disparity_map(i,j) = -1; continue; end
       
       idx = find(responses == max(responses(:)));
       [~,closestIndex] = min(abs(idx-j));
       disparity_map(i,j) = abs(idx(closestIndex) - j);
    end
end
end