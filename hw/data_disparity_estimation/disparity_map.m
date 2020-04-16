function dm = disparity_map(matching_pts, target_img, threshold)
    [x,y,~] = size(target_img);
    dm = zeros(x,y);
    for i=1:x
       for j = 1:y
           pts = matching_pts(i,j,:);
           yp = pts(:,:,2);
           disparity = abs(j - yp);
           if disparity > threshold
               dm(i,j) = 0;
           else
               dm(i,j) = disparity;
           end
       end
    end
end