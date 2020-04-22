function disparity_map = correlation_matching(img1, img2, template_width, distance_threshold)
[x1,y1,~] = size(img1);
[x2,y2,~] = size(img2);
disparity_map = zeros(x1,y1) - 1;

for i = 1:x1
	for j = 1:y1
        % crop a patch around the center
        [img_tmp, central_idx] = get_img_region(img1,i,j,template_width); % both 2d arrays
        j_range = j - distance_threshold:1:j+distance_threshold;
        if j-distance_threshold < 1, j_range = 1:1:j+distance_threshold; end
        if j+distance_threshold > y2, j_range = j - distance_threshold:1:y2; end
        cur_best_response = inf;
        cur_pixel = [i j];
        ischanged = 0;
        % find best match
        for j_idx = 1:length(j_range)
            j2 = j_range(j_idx);
            patch = get_patch(img2,i,j2,x2,y2,central_idx);
            [val, flag] = compare_patch(img_tmp, patch, cur_best_response);

            if flag == 1 && j2 ~= j
                ischanged = 1;
                cur_best_response = val;
                cur_pixel = [i j2];
            end   
        end
        
        if ischanged == 1
            disparity_map(i,j) = abs(j - cur_pixel(2));
        end
	end
end
disparity_map = fix_disparity_map(disparity_map,2);
end

function dm = fix_disparity_map(dm,width)
[x,y] = size(dm);
for i = 1:x
    for j = 1:y
        [pixel_neighbourhood, ~] = get_img_region(dm,i,j,width);
        if dm(i,j) == -1 || dm(i,j) == 0
            dm(i,j) = mean(pixel_neighbourhood(pixel_neighbourhood ~= -1 & pixel_neighbourhood ~= 0));
        else
            t = pixel_neighbourhood(pixel_neighbourhood ~= -1);
            dm(i,j) = (dm(i,j) + sum(t,'all')) / (length(t) + 1);
        end
        
%         qt = 0; count = 0;
%         if i-1 >= 1 && j-1 > 0 && dm(i-1,j-1) ~= -1, qt = qt + dm(i-1,j-1); count = count + 1; end
%         if i-1 >= 1 && dm(i-1,j) ~= -1, qt = qt + dm(i-1,j); count = count + 1; end
%         if i-1 >= 1 && j+1 <= y && dm(i-1,j+1) ~= -1, qt = qt + dm(i-1,j+1); count = count + 1; end
%         if j-1 >= 1 && dm(i,j-1) ~= -1, qt = qt + dm(i,j-1); count = count + 1; end
%         if j+1 <= y && dm(i,j+1) ~= -1, qt = qt + dm(i,j+1); count = count + 1; end
%         if j-1 >= 1 && i+1 <= x && dm(i+1,j-1) ~= -1, qt = qt + dm(i+1,j-1); count = count + 1; end
%         if i+1 <= x && dm(i+1,j) ~= -1, qt = qt + dm(i+1,j); count = count + 1; end
%         if i+1 <= x && j+1 <= y && dm(i+1,j+1) ~= -1, qt = qt + dm(i+1,j+1); count = count + 1; end
%         
%         if count == 0 && dm(i,j) == -1
%             continue;
%         elseif dm(i,j) == -1
%             dm(i,j) = qt;
%         elseif abs((qt / count) - dm(i,j)) > dm(i,j)/8   
%             dm(i,j) = (dm(i,j) + qt) / (count+1);
%         end
    end
end
dm(dm == 0) = mean(dm(dm ~= 0));
end

function [quantity, flag] = compare_patch(img_tmp, patch, cur_best_response)
if isempty(patch) == 1
    quantity = inf;
    flag = 0;
else
    % quantity = sqrt(sum((img_tmp - patch).^2, 'all'));
    quantity = sqrt(sum((img_tmp - patch).^2, 'all'));
    if quantity < cur_best_response
        flag = 1;
    else
        flag = 0;
    end
end
end

function patch = get_patch(img,x,y,ximg,yimg,tmp_idx)
patch_idx = [x + tmp_idx(:,1) y + tmp_idx(:,2)];
if isempty(find(patch_idx(:,1) < 1, 1)) == 1 && isempty(find(patch_idx(:,1) > ximg, 1)) == 1 ...
    && isempty(find(patch_idx(:,2) < 1, 1)) == 1 && isempty(find(patch_idx(:,2) > yimg, 1)) == 1
    r = img(:,:,1);
    rtmp = double(r(sub2ind([ximg yimg], patch_idx(:,1), patch_idx(:,2))));
    % rtmp = rtmp - min(rtmp(:));
    g = img(:,:,2);
    gtmp = double(g(sub2ind([ximg yimg], patch_idx(:,1), patch_idx(:,2))));
    % gtmp = gtmp - min(gtmp(:));
    b = img(:,:,3);
    btmp = double(b(sub2ind([ximg yimg], patch_idx(:,1), patch_idx(:,2))));
    % btmp = btmp - min(btmp(:));
    patch = [rtmp gtmp btmp];
else
    patch = [];
end
end

function [img_tmp, central_idx] = get_img_region(img,x,y,w)
[ximg,yimg,numchannels] = size(img);
mask = ones(2*w+1,2*w+1);
if x-w < 1
    amount = 1 - (x-w);
    if amount > 0
        mask(1:1:amount,:) = 0;
    end
end

if x+w > ximg
    amount = (x+w) - ximg;
    if amount > 0
        mask(2*w+1:-1:2*w+1-amount + 1,:) = 0;
    end
end

if y-w < 1
    amount = 1 - (y-w);
    if amount > 0
        mask(:,1:1:amount) = 0;
    end
end

if y+w > yimg 
    amount = (y+w) - yimg;
    if amount > 0
        mask(:,2*w+1:-1:2*w+1-amount + 1) = 0;
    end
end

[xidx,yidx] = find(mask == 1);
central_idx = [xidx - min(xidx) yidx - min(yidx)];

if numchannels == 3
    r = img(:,:,1);
    rtmp = double(r(sub2ind([ximg yimg], central_idx(:,1) + 1, central_idx(:,2) + 1)));
    % rtmp = rtmp - mean(rtmp(:));
    g = img(:,:,2);
    gtmp = double(g(sub2ind([ximg yimg], central_idx(:,1) + 1, central_idx(:,2) + 1)));
    % gtmp = gtmp - mean(gtmp(:));
    b = img(:,:,3);
    btmp = double(b(sub2ind([ximg yimg], central_idx(:,1) + 1, central_idx(:,2) + 1)));
    % btmp = btmp - mean(btmp(:));
    img_tmp = [rtmp gtmp btmp];
else
    img_tmp = double(img(sub2ind([ximg yimg], central_idx(:,1) + 1, central_idx(:,2) + 1)));
end
end