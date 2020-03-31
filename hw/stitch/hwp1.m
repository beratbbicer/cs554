image_sets = {'im1', 'im2'};
data_loc = 'data_image_stitching/';
ext = '.png';

for i = 1:size(image_sets, 2)
orig_img = imread([data_loc image_sets{i} ext]);
rgb_img = orig_img;
SIFT = SiftMain(orig_img);
show_SIFT_results(SIFT, rgb_img)
end

function show_SIFT_results(SIFT, rgb_img)
DominantOrientation = SIFT.DO;
DiffMinMaxMapNoEdge = SIFT.MAP;
img = SIFT.img;
imgWithPoints = img*0.5;
for i = 1:3
for j = 1:3
for k = 1:2
if isempty(DominantOrientation{i,j})
continue
end
Map = DiffMinMaxMapNoEdge{i,j};
SSCindx = find(Map); % Scale-Space-Corner Index
for ssc = 1:length(SSCindx)
    [Row,Col] = ind2sub([size(Map,1),size(Map,2)], SSCindx(ssc));
    Row = Row*2^(i-1)+1; Col = Col*2^(i-1)+1; % offset = [1,1];
    if (Row > size(img,1)) || (Col > size(img,2))
        continue
    end
    imgWithPoints(Row,Col) = 1;
    rgb_img(Row,Col,1) = 255;
    rgb_img(Row,Col,2) = 0;
    rgb_img(Row,Col,3) = 0;
end
end
end
end
figure,imshow(uint8(imgWithPoints*255));
figure;imshow(rgb_img)
end