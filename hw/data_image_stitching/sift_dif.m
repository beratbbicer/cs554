[matching_idx1, matching_idx2] = sift_matching(d1, d2, 1);
mpts1 = f1(matching_idx1, 1:2); mpts2 = f2(matching_idx2, 1:2);
[mpts1, mpts2] = clean_matches(mpts1, mpts2);

subplot(2, 1, 1)
show_match(mpts1, mpts2, IMG);
title('SIFT Matches, T = 1')

[matching_idx1, matching_idx2] = sift_matching(d1, d2, 2);
mpts1 = f1(matching_idx1, 1:2); mpts2 = f2(matching_idx2, 1:2);
[mpts1, mpts2] = clean_matches(mpts1, mpts2);