import os
from stitch import stitch
import cv2


def stitch_all():
    """
    [TO DO]
    1) Stitch all the images located in img_path by your algorithm
    2) Save result image to the save_path
    Notice:
        stitch() function should be used
        Your algorithm should consider multiple images (more than 2)
    """
    img_path = './imgs/' # don't modify
    save_path = './result.jpg' # don't modify

    imgs = sorted(f for f in os.listdir(img_path))
    src = cv2.imread(os.path.join(img_path, imgs[0]))
    for img in imgs[1:]:
        dst = cv2.imread(f"{img_path}/{img}")
        src = stitch(src, dst)


    cv2.imwrite(save_path, src)
    return src

    ##############
    # write code #
    ##############

    
############# don't modify #############
if __name__ == '__main__':
    stitch_all()
