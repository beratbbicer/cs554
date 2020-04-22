function h = fit_homography(pts1,pts2, iterations, lambda)
 % the more homography iterations the better!
[x, ~] = size(pts1);
h = randn(1,8);
J = zeros(x,8);
residuals = zeros(x,1);
% avg_res = zeros(iterations,1);
for it = 1:iterations
    for i = 1:x
        J(i,:) = jacobian_homography(h,pts1(i,1),pts1(i,2),pts2(i,1),pts2(i,2));
        [x_p,y_p] = get_correspondance(h,pts1(i,1),pts1(i,2));
        residuals(i) = distance_homography(pts2(i,1), pts2(i,2), x_p, y_p);
    end
    % disp(residuals');
    delta_h = pinv(J' * J + randn(8).*(10^-5)) * J' * residuals; 
    h = h - delta_h' * lambda;
    % avg_res(it) = mean(residuals);
end
% figure();
% plot(1:1:iterations,avg_res);
end
