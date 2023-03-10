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

- **blue points**: inliers

- **red points**: outliers

- **red lines**: epipolar lines

- **green text next to points**: approximate pixel distance

### **Remove observations that do not satisfy the epipolar constraint based on the stereo extrinsic.**

When dealing with dynamic objects, it's important to make accurate observations since points on such objects can't be refined through repeated observations. OpenVINS, unfortunately, doesn't verify epipolar constraints for stereo matching, which can result in outliers. To address this, I check for epipolar constraints and remove outliers.

While epipolar lines are typically constructed in the horizontal direction for horizontally-arranged stereo cameras, I **use a fundamental matrix** to construct epipolar lines to **increase accuracy** and **not lose generality**. I then detect outliers by comparing the distance between the calculated epipolar lines and the tracked points in the undistorted, normalized plane. Since distances are usually dominated by the vertical direction, the outlier detection threshold is scaled down by a factor of `fy`, which is the vertical scaling factor.

If you want to compute a **more meaningful threshold**, you can take into account the `angle of the epipolar line`, as well as the `fx` and `fy` values.

The stereo cameras are assumed to be time synchronised, as assumed by OpenVINS.

## Extracting dynamic points using L2 in 3D and reprojection errors
<img src = "https://user-images.githubusercontent.com/72921481/220283402-932e872c-7f6a-4a10-a6ea-b5f9abe0aa8a.png" width="70%" height="70%">

- **white points**: observations from the previous frame

- **red points**: observations in the current frame

- **pink points**: predictions from the current frame to the previous frame

- **points with red text**: static points extracted by RANSAC like dynamic points, but detected by the algorithm.

- **points with green text**: dynamic points that are being tracked.

- **points with green text**: L2 in pixels (reprojection error) and L2 in 3D.


### **Generously extract dynamic points by analyzing the error between prediction and observation.**

To extract dynamic points, I apply the following three conditions to the points where KLT (Kanade-Lucas-Tomasi) algorithm succeeds:
- The point is not too far away.
- The difference between prediction and observation in 3D is above a certain threshold.
- The reprojection error in the undistorted normalized plane is above a certain threshold.

I have the 3D point position at time `t-1`, the 3D point position at time `t`, and the transformation between time `t-1` and `t`. If the point is static, applying the transformation to the 3D point position at `t-1` will result in the same 3D point position at `t`. In other words, the prediction will match the observation. I compare the prediction and observation in both 2D and 3D.

Although it's possible to detect dynamic objects by applying RANSAC on the fundamental matrix at simple `t-1` and `t`, there are drawbacks to using RANSAC:
- The consensus of the fundamental matrix is based only on the distance to the epipolar line, i.e., the epipolar constraint.
- RANSAC relies on probabilities.

**The consensus of the fundamental matrix** means that the distance of the point to the epipolar line is below a threshold. The problem with this is that there are **additional degrees of freedom in the direction of the epipolar line**, which can lead to inaccuracies. In contrast, **reprojection error and the comparison of predictions and observations** in 3D involve a **point-to-point comparison**, which eliminates any remaining degrees of freedom.

I penalize the z direction in the 3D comparison to compensate for the inaccurate depth of the stereo camera.

## Graph construction using k-d tree
<img src = "https://user-images.githubusercontent.com/72921481/220625831-a154603b-4e85-4cbb-9c41-88596bc39952.png" width="70%" height="70%">

- **white points**: dynamic points

- **red connections**: graphs constructed with nearby nodes, but rejected due to insufficient number of nodes.

- **green connections**: graphs to be used (not fully connected in reality, unlike in the picture)

### **Quickly extract neighboring points using k-d tree and create a graph based on the extracted points.**

In the Multimotion Visual Odometry (MVO) paper, the graph is made by calculating the maximum distance between the pixel dimensions of dynamic points within a sliding window. However, for only the current points, I construct the graph in 3D because:

- Simple pixel distances unfairly penalize points that are close together while rewarding those far apart.
- Using the maximum distance between dynamic points within a sliding window is not a simple Euclidean coordinate system, and it becomes computationally expensive for a large number of dynamic points.

To address these issues, I construct the graph by using a k-d tree in 3D for the current points. To construct the graph, I extract k neighboring points within a specific range for each dynamic point. Then, I traverse the constructed graph using DFS to reject graphs with fewer nodes and make the elements easier to process.

This algorithm has a time complexity of `O(N log N)`, where `N` is the number of dynamic points.

## Providing initial labels and removing outlier graphs using RANSAC and registration
<img src = "https://user-images.githubusercontent.com/72921481/221565464-f8e7797a-ee8b-4aee-840d-12fe6725db25.png" width="70%" height="70%">

- **borderless white points** (index 32, 33, ...): nodes in the outlier graphs

- **bounded points** (index 36, 21, 41, ...): nodes in the inline graphs

- **bounded white points** (index 45): outliers in the inlier graphs

- **graphs with different colors** (like the one in the white car in the bottom right): graphs with different labels

### **Apply SVD registration to the sampled nodes to find rigid body transformations, and RANSAC to assign initial labels to the nodes and remove outlier graphs.**

I apply SVD registration by sampling 3 points on the graph to obtain a rigid body transformation. I remove nodes that consensus on the proposed transformation and repeat RANSAC until the graph contains only outliers. The initial labels are provided by RANSAC.

Since I know the point correspondences, I use SVD-based registration to find the rigid body transformation quickly and without iteration. Although it's possible to find the transformation with more than 4 points, I generally **use a minimum sampling of 3 points** because **the probability of outliers being included increases dramatically with more points.**

I **remove graphs that consist only of outliers**. For example, if I have a graph with only 3 points and I compute a transformation based on 3 points, I assume that all points are outliers if they have 2 or fewer consensus. This is a case where **SVD suggests** a transformation **when the motion is not a rigid body motion**. For instance, if three points in space are contracting or expanding.

Consensus is the difference in 3D between the point after applying the transformation to the point at time `t-1` and the observed point at time `t`.

## Rejecting static rigid body transformations using adaptive thresholds
<img src = "https://user-images.githubusercontent.com/72921481/224355171-81051050-c021-47af-a490-e1b389c61525.png" width="70%" height="70%">

- **colored points**: rigid body transformations proposed by RANSAC

- **green circles**: dynamic rigid body transformations extracted by adaptive thresholds

- **green text**: L2 error of the rigid body transform in 2D and 3D / thresholds

### **Secondary extracts dynamic rigid body transformations with strict policies.**

In my previous attempts at dynamic point extractions, I used a lenient policy that could mistakenly identify dynamic points as static points due to observation errors. While it achieved the intended goal, it also resulted in extracting a large number of static points. Therefore, I have decided to refine my approach once again.

To determine the dynamic points, I use a method similar to the previous one. I compare the two predictions separately in both 2D and 3D. This comparison is only done for points that consensus the rigid body transformation suggested by RANSAC. Here are the two predictions:

- The first prediction is performed by assuming that the point is dynamic. To get a refined transformation, I use SVD registration on the points at time `t-1` and `t`. Then, I apply the refined transformation to the points at time `t-1`.

- The second prediction assumes that the points are static. I apply the camera's transformation to the points at time `t-1`.

If the difference between these two predictions is larger than a certain threshold, I assume that the points are dynamic.

When **dealing with a large number of input points**, I simplify the SVD registration process by selecting **the k points with the smallest L2 values based on the rigid body transformation suggested by RANSAC**. This helps to limit the complexity of the registration.

To determine whether the points are dynamic or not, I calculate the difference for all points and divide it by the number of points. Then, I compare this value to the threshold in both 3D and 2D. Here are the details:

- Threshold in 3D: $\alpha + {{\beta}\over{N}}$.
- Threshold in 2D: ${{\gamma}\over{\bar{z}}} + {{\delta}\over{N}}$
- $\alpha, \beta, \gamma, \delta$: constants
- $N$ : number of points
- $\bar{z}$: average depth of the points

When calculating the L2 difference in 3D, I also include a penalty in the z direction. This is to compensate for the depth inaccuracy of stereo cameras. For the thresholds, **the second term helps to reduce uncertainty**, as using fewer points makes it unlikely that the mean error is non-zero based on **the central limit theorem**.

For 2D comparisons, **the first term compensates** for the disadvantage of comparing differences in 2D for **points closer to the camera**. This is because the **image projection** of an object is **inversely proportional to its depth**. The second term serves the same purpose as in the 3D case.

I have imposed stricter constraints for dynamic point extraction, as the following reasons suggest:

- Instead of analyzing individual points, I grouped them to analyze L2, which is less susceptible to errors.
- Since I **assume a rigid transformation** and have **information at both time `t-1` and `t`**, the **accuracy is higher** compared to comparing the **observation at time `t`** to the prediction that applies the camera's transformation to the point at `t-1`.
