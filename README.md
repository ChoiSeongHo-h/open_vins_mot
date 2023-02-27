# open_vins_mot
**in progress...**    
## Environmental
**OpenCV** 4.2   
**Ceres** 2.1   
**Eigen** 3.3.7   
**Ubuntu** 20.04   
**ROS** Noetic with **PCL** 1.10.0   
**OpenVINS** 2.6.2   
Testing on the **KAIST Complex Urban Dataset**   

## Rejecting outliers using stereo extrinsics
<img src = "https://user-images.githubusercontent.com/72921481/219631769-70dda35a-7cfb-4231-84f3-c2c070bbc06b.png" width="70%" height="70%">

- blue points: inliers

- red points: outliers

- red lines: epipolar lines

- green text next to points: approximate pixel distance

### **Remove observations that do not satisfy the epipolar constraint based on the stereo extrinsic.**

When dealing with dynamic objects, it's important to make accurate observations since points on such objects can't be refined through repeated observations. OpenVINS, unfortunately, doesn't verify epipolar constraints for stereo matching, which can result in outliers. To address this, I check for epipolar constraints and remove outliers.

While epipolar lines are typically constructed in the horizontal direction for horizontally-arranged stereo cameras, I use a fundamental matrix to construct epipolar lines to increase accuracy and not lose generality.. I then detect outliers by comparing the distance between the calculated epipolar lines and the tracked points in the undistorted, normalized plane. Since distances are usually dominated by the vertical direction, the outlier detection threshold is scaled down by a factor of `fy`, which is the vertical scaling factor.

If you want to compute a more meaningful threshold, you can take into account the angle of the epipolar line, as well as the `fx` and `fy` values.

The stereo cameras are assumed to be time synchronised, as assumed by OpenVINS.

## Extracting dynamic points using reprojection errors and L2 in 3D
<img src = "https://user-images.githubusercontent.com/72921481/220283402-932e872c-7f6a-4a10-a6ea-b5f9abe0aa8a.png" width="70%" height="70%">

- white points: observations from the previous frame

- red points: observations in the current frame

- pink points: predictions from the current frame to the previous frame

- points with red text: static points extracted by RANSAC like dynamic points, but detected by the algorithm.

- points with green text: dynamic points that are being tracked.

- points with green text: L2 in pixels (reprojection error) and L2 in 3D.


### **Extract dynamic points by analyzing the error between prediction and observation.**

To extract dynamic points, I apply the following three conditions to the points where KLT (Kanade-Lucas-Tomasi) algorithm succeeds:
1. The point is not too far away.
2. The difference between prediction and observation in 3D is above a certain threshold.
3. The reprojection error in the undistorted normalized plane is above a certain threshold.

I have the 3D point position at time t-1, the 3D point position at time t, and the transformation between time t-1 and t. If the point is static, applying the transformation to the 3D point position at t-1 will result in the same 3D point position at t. In other words, the prediction will match the observation. I compare the prediction and observation in both 2D and 3D.

Although it's possible to detect dynamic objects by applying RANSAC on the fundamental matrix at simple t-1 and t, there are drawbacks to using RANSAC:
1. The consensus of the fundamental matrix is based only on the distance to the epipolar line, i.e., the epipolar constraint.
2. RANSAC relies on probabilities.

**The consensus of the fundamental matrix** means that the distance of the point to the epipolar line is below a threshold. The problem with this is that there are **additional degrees of freedom in the direction of the epipolar line**, which can lead to inaccuracies. In contrast, **reprojection error and the comparison of predictions and observations** in 3D involve a **point-to-point comparison**, which eliminates any remaining degrees of freedom.

It imposes a penalty in the z direction to compensate for the inaccurate depth of the stereo camera.

## Graph construction using k-d tree
<img src = "https://user-images.githubusercontent.com/72921481/220625831-a154603b-4e85-4cbb-9c41-88596bc39952.png" width="70%" height="70%">

- white points: dynamic points

- red connections: graphs constructed with nearby nodes, but rejected due to insufficient number of nodes.

- green connections: graphs to be used (not fully connected in reality, unlike in the picture)

### **Quickly extract neighboring points using k-d tree and create a graph based on the extracted points.**

In the Multimotion Visual Odometry (MVO) paper, the graph is made by calculating the maximum distance between the pixel dimensions of dynamic points within a sliding window. However, for only the current input, I construct the graph in 3D because:

1. Simple pixel distances unfairly penalize points that are close together while rewarding those far apart.
2. Using the maximum distance between dynamic points within a sliding window is not a simple Euclidean coordinate system, and it becomes computationally expensive for a large number of dynamic points.

To address these issues, I construct the graph by using a k-d tree in 3D for the current input. To construct the graph, I extract k neighboring points within a specific range for each dynamic point. Then, I traverse the constructed graph using DFS to reject graphs with fewer nodes and make the elements easier to process.

This algorithm has a time complexity of O(N log N), where N is the number of dynamic points.

## Providing initial labels and removing outlier graphs using RANSAC and ICP
<img src = "https://user-images.githubusercontent.com/72921481/221565464-f8e7797a-ee8b-4aee-840d-12fe6725db25.png" width="70%" height="70%">

- Borderless White Points (index 32, 33, ...): Nodes in the Outlier Graphs

- Bounded Points (index 36, 21, 41, ...): Nodes in the Inline Graphs

- Bounded White Points (index 45): Outliers in the Inlier Graphs

- Graphs with different colors (like the one in the white car in the bottom right): Graphs with different labels

### **Apply ICP to the sampled nodes to find transformations, and RANSAC to assign initial labels to the nodes and remove outlier graphs.**

I apply ICP by sampling 3 points on the graph to obtain a transformation. I remove nodes that consensus on the proposed transformation and repeat RANSAC until the graph contains only outliers. The initial labels are provided by RANSAC.

Since I know the point correspondences, I use SVD-based ICP to find the transformation quickly and without iteration. Although it's possible to find the transformation with more than 4 points, I generally **use a minimum sampling of 3 points** because **the probability of outliers being included increases dramatically with more points.**

I remove graphs that consist only of outliers. For example, if I have a graph with only 3 points and I compute a transformation based on 3 points, I assume that all points are outliers if they have 2 or fewer consensus. This is a case where ICP suggests a transformation when the motion is not a rigid body motion. For instance, if three points in space are contracting or expanding.

Consensus is the difference in 3D between the point after applying the transformation to the point at time t-1 and the observed point at time t.
