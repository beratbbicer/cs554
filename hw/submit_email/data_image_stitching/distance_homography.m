function d = distance_homography(x,y,x_p,y_p)
    d = sqrt((x_p - x)^2 + (y_p - y)^2);
end