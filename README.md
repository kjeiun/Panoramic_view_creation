# Panoramic_view_creation
This project allows you to create a panoramic view by selecting a corresponding point. Using this correspondences, warp the src img by backward warping and this allows you to make panoramic img using stitch function.

get_coord.py

functions: get_coord




homography.py


functions:


get_homography(src, dst)




ransac(src_points, dst_points, iteration, threshold)


stitch.py




functions:


warp(image, H, target_w, target_h)


stitch(src_img, dst_img)




main.py


functions:


stitch_all()




get_coord(ssrc_img, ddst_img): This function allows you to choose correspondences between two images

    Args:
        ssrc_img: np.ndarray with shape (src_h,src_w,c)
        ddst_img: np.ndarray with shape (dst_h,dst_w,c)
    Returns:
        src_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        dst_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        (i)th correspondence: src_points[i] <-> dst_points[i]
        
<img width="396" alt="image" src="https://github.com/kjeiun/Panoramic_view_creation/assets/87067659/38300c9f-4745-4e6f-9197-7d692084613d">

    [Usage guideline]
    - Two windows (left for src image and right for dst image) are created
    1) Click a point at src image and press enter
    2) Click the corresponding point at dst image and press enter
    - One point correspondence is registered
    3) Repeat multiple times as you want
    4) After indicating multiple point correspondences,
        press enter twice in a row (started by src image) to exit
 
 get_homography(src_points, dst_points): Input of this function can be generated by get_coord function. The output is homography matrix between src_img and dst_img by your own correspondences
 
      Args:
        src_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        dst_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        (i)th correspondence: src_points[i] <-> dst_points[i]
        
      Returns:
          H: Homography matrix with shape (3,3)
          H[2,2] = 1
 Ransac(src_points, dst_points, iteration, threshold)
 :This function allows you to find the best homography among all possibles. The best homography implies that this H has the biggest number of inliers
     
     Args:
        src_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        dst_points: np.ndarray with shape (num_points,2), 2=> xy(or wh) axis
        (i)th correspondence: src_points[i] <-> dst_points[i]
        iteration : the number of iterations to find best homography
        threshold : If the error ( input dst_points and calculated dst_points using homography) is less than threshold, then this point becomes inlier.
        
        Output: 
          H: Best Homography matrix with shape(3,3) , H[2,2] = 1
        
  warp(image, H, target_w, target_h): Warp src_img (using homography matrix H) to the target plane with a window size=(target_h,target_w)
      
      Args:
        src_img: np.ndarray with shape (src_h,src_w,c)
        H: np.ndarray with shape (3,3)
        target_h: int
        target_w: int
    Returns:
        Warped src image
        np.ndarray with shape (target_h,target_w,c)
   
 stitch(src_img, dst_img): 
 
    1)Manually indicate corresponding points
    2) Find src_img -> dst_img homography matrix
    3) Warp src_img and stitch with dst_img 

    Args:
        src_img: np.ndarray with shape (src_h,src_w,c)
        dst_img: np.ndarray with shape (dst_h,dst_w,c)
    Returns:
        src,dst stitched image. np.ndarray with shape (target_h,target_w,c)
    Notice:
        get_homography() or ransac() function should be used for computing homography
        warp() function should be used for warping an image

  [Example]

<img width="436" alt="image" src="https://github.com/kjeiun/Panoramic_view_creation/assets/87067659/116c2199-e914-4c8c-8826-bf211767e089">


<img width="356" alt="image" src="https://github.com/kjeiun/Panoramic_view_creation/assets/87067659/ee81cda0-a745-4797-93ed-b8ddea9a6919">



#stitched boudary



<img width="176" alt="image" src="https://github.com/kjeiun/Panoramic_view_creation/assets/87067659/dea8da32-7bfa-4144-abbe-6db3211d02a7">



<img width="440" alt="image" src="https://github.com/kjeiun/Panoramic_view_creation/assets/87067659/c85b6e1c-6ee0-4189-9bf3-317499162330">

