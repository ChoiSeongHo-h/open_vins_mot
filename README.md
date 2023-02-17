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

## Rejecting outliers in stereo matching
<img src = "https://user-images.githubusercontent.com/72921481/219631769-70dda35a-7cfb-4231-84f3-c2c070bbc06b.png" width="70%" height="70%">
Blue circles: inliers, red circles: outliers, red lines: epipolar lines, green numbers: approximate pixel distance
<br/>
<br/>
Since dynamic objects are not stationary, the distance to the object must be obtained immediately at one time. Also, the points on a dynamic object at a specific time cannot be refined through repeated observations, so accurate observations are required.    
Unfortunately, OpenVINS does not perform epipolar constraint verification for stereo matching. This implementation checks for epipolar constraints and removes outliers.    
<br/>
<br/>
Since stereo cameras are usually arranged horizontally, the epipolar lines are represented in the horizontal direction. However, without loss of generality, and to improve accuracy, I use the extrinsic between the two cameras to construct the fundamental matrix. Outliers are detected by the distance between the calculated epipolar lines and the points tracked by the optical flow. The distance calculation is performed in the undistorted, normalized plane due to the real precision.
<br/>
<br/>
As mentioned above, epipolar lines are typically formed in the horizontal direction, so distances are typically dominated by the vertical direction. On the other hand, the image is scaled down by `fy` in the vertical direction, so the threshold for outlier detection should also be scaled down by a factor of `fy`. If you want a more meaningful threshold, you can consider the threshold scaling factor by considering the angle of the epipolar line, `fx`, and `fy`.
<br/>
<br/>
The stereo cameras are assumed to be time synchronised, as assumed by OpenVINS.
