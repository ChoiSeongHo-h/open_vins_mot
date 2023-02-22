# open_vins_mot
**in progress...**    
## Environmental
**OpenCV** 4.2   
**Ceres** 2.1   
**Eigen** 3.3.7   
**Ubuntu** 20.04   
**ROS** Noetic with **PCL**   
**OpenVINS** 2.6.2   
Testing on the **KAIST Complex Urban Dataset**   

## Rejecting outliers using stereo extrinsics
<img src = "https://user-images.githubusercontent.com/72921481/219631769-70dda35a-7cfb-4231-84f3-c2c070bbc06b.png" width="70%" height="70%">

- blue points: inliers

- red points: outliers

- red lines: epipolar lines

- green text next to points: approximate pixel distance

When dealing with dynamic objects, it's important to make accurate observations since points on such objects can't be refined through repeated observations. OpenVINS, unfortunately, doesn't verify epipolar constraints for stereo matching, which can result in outliers. To address this, I check for epipolar constraints and remove outliers.

While epipolar lines are typically constructed in the horizontal direction for horizontally-arranged stereo cameras, I use a fundamental matrix to construct epipolar lines for better accuracy. I then detect outliers by comparing the distance between the calculated epipolar lines and the tracked points in the undistorted, normalized plane. Since distances are usually dominated by the vertical direction, the outlier detection threshold is scaled down by a factor of fy, which is the vertical scaling factor.

If you want to compute a more meaningful threshold, you can take into account the angle of the epipolar line, as well as the fx and fy values.

The stereo cameras are assumed to be time synchronised, as assumed by OpenVINS.

## Extracting dynamic points using reprojection errors and L2 in 3D
<img src = "https://user-images.githubusercontent.com/72921481/220283402-932e872c-7f6a-4a10-a6ea-b5f9abe0aa8a.png" width="70%" height="70%">

- white points: observations from the previous frame

- red points: observations in the current frame

- pink points: predictions from the current frame to the previous frame

- points with red text: static points extracted by RANSAC like dynamic points, but detected by the algorithm.

- points with green text: dynamic points that are being tracked.

- points with green text: L2 in 3D and L2 in pixels (reprojection error).


OpenVINS removes dynamic points using RANSAC on the previous and current frames. Since the points rejected by RANSAC are usually dynamic, I want to keep track of them.

Using a generous RANSAC threshold will not extract dynamic points well, so I use a harsh threshold. However, this also results in the extraction of static points, so I need to remove them. I use three criteria to identify dynamic points:

1. The point isn't too far away.

2. The L2 penalized value in the z direction in 3D Euclidean space is below a certain threshold.

3. The reprojection error in the undistorted and normalized plane is below a certain threshold.

In general, **static points to be removed** (rejected at the same time as dynamic points in RANSAC) **are extracted from areas close to the camera**. **Since the fundamental matrix proposed by RANSAC is thresholded at the pixel dimension, close points are more sensitive to this proposal than distant points.** Furthermore, it is harshly thresholded and thus very much extracted. To solve this problem, I don't rely on RANSAC's simple epipolar constraints, but instead use the reprojection error and 3D L2 error. The reprojection error measures the difference between the prediction through motion and the observation, while the 3D L2 error penalizes the z-direction to compensate for the stereo camera's inaccurate depth.

## Graph construction using k-d tree
<img src = "https://user-images.githubusercontent.com/72921481/220625831-a154603b-4e85-4cbb-9c41-88596bc39952.png" width="70%" height="70%">

- white dots: dynamic points

- red connections: graphs constructed with nearby nodes, but rejected due to insufficient number of nodes.

- green connections: graphs to be used ( not fully connected in reality, unlike in the picture)

Before labeling the dynamic points, I create a graph of neighboring points.

In the Multimotion Visual Odometry (MVO) paper, the graph is made by calculating the maximum distance between the pixel dimensions of dynamic points within a sliding window. However, for the current input, I construct the graph in 3D because:

1. Simple pixel distances unfairly penalize points that are close together while rewarding those far apart.
2. Using the maximum distance between dynamic points within a sliding window is not a simple Euclidean coordinate system, and it becomes computationally expensive for a large number of dynamic points.

To address these issues, I construct the graph by using a k-d tree in 3D for the current input. To construct the graph, I extract k neighboring points within a specific range for each dynamic point. Then, I traverse the constructed graph using DFS to reject graphs with fewer nodes and make the elements easier to process.

This algorithm has a time complexity of O(N log N), where N is the number of dynamic points.
