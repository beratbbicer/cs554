function [rimg, xmin, ymin] = rectification(h, img)
[x,y,~] = size(img);
[xp_tl, yp_tl] = get_correspondance(h,1,1);
[xp_tr, yp_tr] = get_correspondance(h,1,y);
[xp_bl, yp_bl] = get_correspondance(h,x,1);
[xp_br, yp_br] = get_correspondance(h,x,y);
xmin = min([xp_tl xp_tr xp_bl xp_br]);
xmax = max([xp_tl xp_tr xp_bl xp_br]);
ymin = min([yp_tl yp_tr yp_bl yp_br]);
ymax = max([yp_tl yp_tr yp_bl yp_br]);

xlen = ceil(xmax - xmin + 1);
ylen = ceil(ymax - ymin + 1);
rimg = uint8(zeros([xlen, ylen, 3]));

for i = 1:x % row
    for j = 1:y % column
        % disp([i j]);
        [xp,yp] = get_correspondance(h,i,j);
        xp = ceil(xp - xmin + 1);
        yp = ceil(yp - ymin + 1);
        rimg(xp,yp,:) = img(i,j,:);
    end
end

for i = 1:xlen % row
    for j = 1:ylen % column
        if rimg(i, j, 1) == 0
            xint = [i-1 i i+1];
            yint =  [j-1 j j+1];

            if i == 1
                xint = [i i+1];
            elseif i == xlen
                xint = [i-1 i];
            end

            if j == 1
                yint = [j j+1];
            elseif j == ylen
                yint = [j-1 j];
            end

            idx = find(rimg(xint, yint, 1) ~= 0);

            if length(idx) > 4
                rpatch = rimg(xint, yint, 1);
                gpatch = rimg(xint, yint, 2);
                bpatch = rimg(xint, yint, 3);
                rimg(i, j, 1) = uint8(mean(rpatch(idx)));
                rimg(i,j,2) = uint8(mean(gpatch(idx)));
                rimg(i,j,3) = uint8(mean(bpatch(idx)));
            end
        end
    end
end
end