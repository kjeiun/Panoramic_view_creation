import numpy as np
from get_coord import get_coord
from homography import get_homography, ransac
import cv2
import math

def warp(image, H, target_w, target_h):
    # 이미지의 크기를 가져옴
    height = image.shape[0]
    width = image.shape[1]

    # 역변환을 위한 그리드 생성
    x, y = np.meshgrid(np.arange(-(target_w//2), target_w-(target_w//2)), np.arange(-(target_h//2), target_h-(target_h//2)))
    x = x.flatten()
    y = y.flatten()
    # 그리드 좌표를 Homogeneous 좌표로 변환
    ones = np.ones_like(x)
    coordinates = np.vstack((x, y, ones))
    # Homography 행렬을 적용하여 dst 좌표 계산
    warped_coordinates = np.dot(np.linalg.inv(H), coordinates)
    warped_coordinates /= warped_coordinates[2, :]

    warped_x = warped_coordinates[0, :].reshape(target_h, target_w)
    warped_y = warped_coordinates[1, :].reshape(target_h, target_w)
    warped_image = np.zeros(target_h*target_w*3).reshape(target_h, target_w,3)

    for X in range(target_w):
        for Y in range(target_h):
            srcY = warped_y[Y,X]
            srcX = warped_x[Y,X]
            if 0<=srcY<height-1 and 0<=srcX<width-1:
                warped_image[Y, X,:] = image[round(srcY),round(srcX),:]
                #Bilinear interpolation code인데, 이걸 쓰면 매우 느려서, 가장 가까운 좌표를 할당하는 방식을 채택함.
                # _X, _Y = math.floor(srcX), math.floor(srcY)
                # X_, Y_ = _X+1, _Y+1
                # d_x, d_y, dx_, dy_ = srcX - _X, srcY-_Y, X_-srcX, Y_-srcY
                # interpolated_value = np.ones(3).reshape(1,1,3)
                # interpolated_value[0, 0, :] = image[_Y,_X, :] * dy_ * dx_ + image[Y_, _X, :] * dy_ * d_x  + image[Y_, X_, :]*d_y*d_x + image[_Y, X_, :]*d_y*dx_
                # warped_image[Y,X,:] = interpolated_value[0,0,:]

    return warped_image.astype(np.uint8)

def stitch(src_img, dst_img):
    src_points , dst_points = get_coord(src_img, dst_img)

    src_width = src_img.shape[1]
    src_height = src_img.shape[0]
    dst_width, dst_height = dst_img.shape[1], dst_img.shape[0]
    # H = ransac(src_points, dst_points)
    H = get_homography(src_points,dst_points)

    target_w, target_h = 2*(src_width+dst_width), 2*(src_height +dst_height)
    warped_img = warp(src_img, H, target_w, target_h)
    for x in range(dst_width):
        for y in range(dst_height):
            warped_img[y+(target_h//2),x+(target_w//2),:] = dst_img[y,x, :]

    return warped_img
