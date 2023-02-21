# open_vins_mot
**in progress...**    
## Environmental
**OpenCV** 4.2   
**Ceres** 2.1   
**Eigen** 3.3.7   
**Ubuntu** 20.04   
**ROS** Noetic   
**OpenVINS** 2.6.2   
Testing on the **KAIST Complex Urban Dataset**   

## Rejecting outliers using stereo extrinsics
<img src = "https://user-images.githubusercontent.com/72921481/219631769-70dda35a-7cfb-4231-84f3-c2c070bbc06b.png" width="70%" height="70%">

- blue points: inliers

- red points: outliers

- red lines: epipolar lines

- green text next to points: approximate pixel distance


Since dynamic objects are not stationary, the distance to the object must be obtained immediately at one time. Also, the points on a dynamic object at a specific time cannot be refined through repeated observations, so accurate observations are required.    
Unfortunately, OpenVINS does not perform epipolar constraint verification for stereo matching. This implementation checks for epipolar constraints and removes outliers.    


Since stereo cameras are usually arranged horizontally, the epipolar lines are represented in the horizontal direction. However, without loss of generality, and to increase accuracy, I do not use simple horizontal epipolar lines, but instead construct the epipolar lines with a fundamental matrix. The fundamental matrix is derived from the stereo camera's extrinsic. Outliers are detected by the distance between the calculated epipolar lines and the points tracked by the optical flow. The distance calculation is performed in the undistorted, normalized plane due to the real precision.


As mentioned above, epipolar lines are typically formed in the horizontal direction, so distances are typically dominated by the vertical direction. On the other hand, the image is scaled down by `fy` in the vertical direction with respect to the normalized plane, so the threshold for outlier detection should also be scaled down by a factor of `fy`. If you want a more meaningful threshold, you can consider the threshold scaling factor by considering the angle of the epipolar line, `fx`, and `fy`.


The stereo cameras are assumed to be time synchronised, as assumed by OpenVINS.

## Extracting dynamic points using reprojection errors and L2 in 3D
<img src = "https://user-images.githubusercontent.com/72921481/220283402-932e872c-7f6a-4a10-a6ea-b5f9abe0aa8a.png" width="70%" height="70%">

- white points: observations from the previous frame

- red points: observations in the current frame

- pink points: predictions from the current frame to the previous frame

- points with red text: static points extracted by RANSAC like dynamic points, but detected by the algorithm.

- points with green text: dynamic points that are being tracked.

- points with green text: L2 in 3D and L2 in pixels (reprojection error).


OpenVINS performs RANSAC on the previous and current frames to remove dynamic points. Since the points rejected by RANSAC are usually dynamic, I want to keep track of them.

Using a generous RANSAC threshold will not extract dynamic points well, so I use a harsh threshold. However, this also results in the extraction of static points, so we need to remove them.

Dynamic points are extracted by the following three criteria.

1. not too far away.

2. L2 penalised in the z direction in 3D Euclidean is below a threshold.

3. the reprojection error in the undistorted and normalized plane is below a threshold.

In general, **static points to be removed** (rejected at the same time as dynamic points in RANSAC) **are extracted from areas close to the camera**. **Since the fundamental matrix proposed by RANSAC is thresholded at the pixel dimension, close points are more sensitive to this proposal than distant points.** Furthermore, it is harshly thresholded and thus very much extracted. Therefore, we depart from RANSAC, which checks for simple epipolar constraints, and use the reprojection error and L2 in 3D. The reprojection error compares the difference between the prediction through motion and the observation. The 3D L2 error penalises the z-direction to compensate for the inaccurate depth of the stereo camera.
