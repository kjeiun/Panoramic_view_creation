import numpy as np
import torch

import torch

#흠 일단 얜 정상적으로 잘 작동함.
def get_homography(src_points, dst_points):
    src_points = torch.tensor(src_points, dtype=torch.float64)
    dst_points = torch.tensor(dst_points, dtype=torch.float64)
    num_points = src_points.shape[0]

    # Define matrix A (x, y)->(x',y')
    A = torch.zeros((2 * num_points, 9), dtype=torch.float64)
    for i in range(2*num_points):
        if i % 2 == 0:
            A[i, 0] = -src_points[int(i/2), 0]
            A[i, 1] = -src_points[int(i/2), 1]
            A[i, 2] = -1
            A[i, 6] = src_points[int(i/2), 0] * dst_points[int(i/2), 0]
            A[i, 7] = src_points[int(i/2), 1] * dst_points[int(i/2), 0]
            A[i, 8] = dst_points[int(i/2), 0]
        elif i % 2 != 0:
            A[i, 3:6] = A[i - 1, 0:3]
            A[i,6] = src_points[int(i/2),0]*dst_points[int(i/2),1]
            A[i,7] = src_points[int(i/2),1]*dst_points[int(i/2),1]
            A[i,8] = dst_points[int(i/2),1]

    _, _, Vh = torch.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    H /= H[2, 2]
    return H.numpy()

def ransac(src_points, dst_points, iteration= 200 , threshold= 0.8):
    H = np.zeros(9).reshape(3,3)
    num_points = len(src_points)
    src_coordinates = np.column_stack(src_points)
    dst_coordinates = np.column_stack(dst_points)
    ones = np.ones(num_points).flatten()
    homo_src = np.vstack((src_coordinates, ones))
    homo_dst = np.vstack((dst_coordinates, ones))
    best_inliers = 0
    for i in range(iteration):
        idx = np.random.choice(num_points, 4, replace=False)
        chosen_src = src_points[idx]
        chosen_dst = dst_points[idx]
        h = get_homography(chosen_src, chosen_dst)
        dst = np.dot(h, homo_src)
        dst /= dst[2,:]
        diff = homo_dst - dst
        l2_diff = diff**2
        error = np.sqrt(l2_diff[0]+ l2_diff[1])
        num_inliers=0
        for j in range(num_points):
            if error[j]<threshold:
                num_inliers += 1
        if num_inliers>best_inliers:
            best_inliers = num_inliers
            H = h

    return H

