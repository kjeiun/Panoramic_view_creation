import numpy as np
import cv2
from tkinter import *


import torch


def draw_circle(img, x, y, color_name):
    h, w = img.shape[:2]
    cr = int(min(h, w) * 0.015)
    color = [0, 0, 255] if color_name == 'red' else [255, 0, 0]  # bgr

    for r in range(cr + 1):
        for dy in range(-r, r + 1):
            if 0 <= y + dy < h:
                dx = r - abs(dy)
                if 0 <= x + dx < w:
                    img[y + dy, x + dx] = color
                if 0 <= x - dx < w:
                    img[y + dy, x - dx] = color


def get_coord(ssrc_img, ddst_img):
    """
    Args:
        ssrc_img: np.ndarray with shape (src_h,src_w,c)
        ddst_img: np.ndarray with shape (dst_h,dst_w,c)
    Returns:
        src_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        dst_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        (i)th correspondence: src_points[i] <-> dst_points[i]

    [Usage guideline]
    - Two windows (left for src image and right for dst image) are created
    1) Click a point at src image and press enter
    2) Click the corresponding point at dst image and press enter
    - One point correspondence is registered
    3) Repeat multiple times as you want
    4) After indicating multiple point correspondences,
        press enter twice in a row (started by src image) to exit

    Notice:
        Only available for a gui envorinment
        You can freely modify the code for your convenience
    """
    h_src, w_src = ssrc_img.shape[:2]
    h_dst, w_dst = ddst_img.shape[:2]

    root = Tk()
    h_monitor = root.winfo_screenheight()
    w_monitor = root.winfo_screenwidth()
    root.destroy()

    scale_h = h_monitor * 0.98 / (h_src + h_dst)
    scale_w = w_monitor * 0.98 / (w_src + w_dst)
    scale = min(scale_h, scale_w)

    src_img = cv2.resize(ssrc_img, dsize=(int(w_src * scale), int(h_src * scale)))
    dst_img = cv2.resize(ddst_img, dsize=(int(w_dst * scale), int(h_dst * scale)))
    cv2.imshow('src', src_img)
    cv2.imshow('dst', dst_img)

    src_points = []
    dst_points = []
    while True:
        src_x, src_y, _, _ = cv2.selectROI('src', src_img, showCrosshair=False)
        cv2.destroyWindow('src')
        draw_circle(src_img, src_x, src_y, 'red')
        cv2.imshow('src', src_img)

        dst_x, dst_y, _, _ = cv2.selectROI('dst', dst_img, showCrosshair=False)
        cv2.destroyWindow('dst')
        draw_circle(dst_img, dst_x, dst_y, 'red')
        cv2.imshow('dst', dst_img)

        if src_x == 0 and src_y == 0 and dst_x == 0 and dst_y == 0:
            cv2.destroyAllWindows()
            break

        cv2.destroyWindow('src')
        draw_circle(src_img, src_x, src_y, 'blue')
        cv2.imshow('src', src_img)
        cv2.destroyWindow('dst')
        draw_circle(dst_img, dst_x, dst_y, 'blue')
        cv2.imshow('dst', dst_img)

        src_x, src_y = int(src_x / scale), int(src_y / scale)
        dst_x, dst_y = int(dst_x / scale), int(dst_y / scale)
        print('src (x,y) = ({}, {}) | dst (x,y) = ({}, {})'.format(
            src_x, src_y, dst_x, dst_y))
        src_points.append([src_x, src_y])
        dst_points.append([dst_x, dst_y])

    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    assert src_points.shape == dst_points.shape and src_points.shape[0] >= 4
    print('src_points:', src_points)
    print('dst_points:', dst_points)
    print()
    return src_points, dst_points


