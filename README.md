# OpenVINS with Moving Object Tracking


https://user-images.githubusercontent.com/72921481/229364790-3a80e7f2-2ea3-4612-9ae1-74700cd98406.mp4

## Overview
![image](https://user-images.githubusercontent.com/72921481/229867588-b62e423f-14c6-4209-9eab-99bd366c3e3a.png)
**The expected effect of this system is as follows:**
- **Track dynamic objects** and assist perception systems **without deep learning**.
- **Improve localization accuracy** by removing biased points.

**This system procedure can be summarized as follows:**
1. Outlier removal using stereo extrinsic.
2. Static point removal using ego motion
3. Neighborhood point graph generation using k-d tree
4. Proposing rigid transformation, removal outliers, and removal static transformation using RANSAC
5. Nonlinear optimization of points and dynamic rigid body transformations
6. Improving tracking and localization accuracy by focusing on dynamic objects

## Environmental
- **Stereo cameras** to determine a moving 3D point at a point in time
- **IMU** to help prevent the system from being dominated by dynamic objects
- **OpenCV** 4.2   
- **Ceres** 2.1   
- **Eigen** 3.3.7   
- **Ubuntu** 20.04   
- **ROS** Noetic with **PCL** 1.10.0   
- **OpenVINS** 2.6.2   
- Testing on the **KAIST Complex Urban Dataset**   

## Rejecting outliers using stereo extrinsics
![image](https://user-images.githubusercontent.com/72921481/229905482-b76a9507-2645-4995-8732-33c5d62e5d2a.png)

![image](https://user-images.githubusercontent.com/72921481/219631769-70dda35a-7cfb-4231-84f3-c2c070bbc06b.png)

- **blue points**: inliers

- **red points**: outliers

- **red lines**: epipolar lines

- **green text next to points**: approximate pixel distance

### **Remove observations that do not satisfy the epipolar constraint based on the stereo extrinsic.**

When dealing with dynamic objects, it's important to make accurate observations since points on such objects can't be refined through repeated observations. OpenVINS, unfortunately, doesn't verify epipolar constraints for stereo matching, which can result in outliers. To address this, I check for epipolar constraints and remove outliers.

While epipolar lines are typically constructed in the horizontal direction for horizontally-arranged stereo cameras, I **use a fundamental matrix** to construct epipolar lines to **increase accuracy** and **not lose generality**. I then detect outliers by comparing the distance between the calculated epipolar lines and the tracked points in the undistorted, normalized plane. Since distances are usually dominated by the vertical direction, the outlier detection threshold is scaled down by a factor of `fy`, which is the vertical scaling factor.

If you want to compute a **more meaningful threshold**, you can take into account the `angle of the epipolar line`, as well as the `fx` and `fy` values.

The stereo cameras are assumed to be time synchronised, as assumed by OpenVINS.

The test is performed at both time `t-1` and `t`.

## Extracting dynamic points using L2 in 3D and reprojection errors
![image](https://user-images.githubusercontent.com/72921481/229908887-6b7899e6-e946-4272-8cc9-cc828c8aceb4.png)

![image](https://user-images.githubusercontent.com/72921481/220283402-932e872c-7f6a-4a10-a6ea-b5f9abe0aa8a.png)

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

I have the 3D point position at time `t-1`, the 3D point position at time `t`, and the transformation between time `t-1` and `t`(ego motion). If the point is static, applying the transformation to the 3D point position at `t-1` will result in the same 3D point position at `t`. In other words, the prediction will match the observation. I compare the prediction and observation in both 2D and 3D.

Although it's possible to detect dynamic objects by applying RANSAC on the fundamental matrix at simple `t-1` and `t`, there are drawbacks to using RANSAC:
- The consensus of the fundamental matrix is based only on the distance to the epipolar line, i.e., the epipolar constraint.
- RANSAC relies on probabilities.

**The consensus of the fundamental matrix** means that the distance of the point to the epipolar line is below a threshold. The problem with this is that there are **additional dof in the direction of the epipolar line**, which can lead to inaccuracies. In contrast, **reprojection error and the comparison of predictions and observations** in 3D involve a **point-to-point comparison**, which eliminates any remaining dof.

I penalize the z direction in the 3D comparison to compensate for the inaccurate depth of the stereo camera.

For consistency, the re-projection error is evaluated for both viewpoints and the greater of the two is used.

## Graph construction using k-d tree
![image](https://user-images.githubusercontent.com/72921481/229906079-dd92f9e8-92e6-41fe-9919-c7821b57dcf5.png)

![image](https://user-images.githubusercontent.com/72921481/220625831-a154603b-4e85-4cbb-9c41-88596bc39952.png)

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
![image](https://user-images.githubusercontent.com/72921481/229906559-1f224ed2-feea-411d-8f71-a09676241215.png)

![image](https://user-images.githubusercontent.com/72921481/221565464-f8e7797a-ee8b-4aee-840d-12fe6725db25.png)

- **borderless white points** (index 32, 33, ...): nodes in the outlier graphs

- **bounded points** (index 36, 21, 41, ...): nodes in the inlier graphs

- **bounded white points** (index 45): outliers in the inlier graphs

- **graphs with different colors** (like the one in the white car in the bottom right): graphs with different labels

### **Apply SVD registration to the sampled nodes to find rigid body transformations, and RANSAC to assign initial labels to the nodes and remove outlier graphs.**

I apply SVD registration by sampling 3 points on the graph to obtain a rigid body transformation. I remove nodes that consensus on the proposed transformation and repeat RANSAC until the graph contains only outliers. The initial labels are provided by RANSAC.

Since I know the point correspondences, I use SVD-based registration to find the rigid body transformation quickly and without iteration. Although it's possible to find the transformation with more than 4 points, I generally **use a minimum sampling of 3 points** because **the probability of outliers being included increases dramatically with more points.**

I **remove graphs that consist only of outliers**. For example, if I have a graph with only 3 points and I compute a transformation based on 3 points, I assume that all points are outliers if they have 2 or fewer consensus. This is a case where **SVD suggests** a transformation **when the motion is not a rigid body motion**. For instance, if three points in space are contracting or expanding.

Consensus is the difference in 3D between the point after applying the transformation to the point at time `t-1` and the observed point at time `t`.

## Rejecting static rigid body transformations using adaptive thresholds

![image](https://user-images.githubusercontent.com/72921481/229908224-031d86f5-6717-4842-9f6c-54e26b2a042e.png)

![image](https://user-images.githubusercontent.com/72921481/224355171-81051050-c021-47af-a490-e1b389c61525.png)

- **colored points**: rigid body transformations proposed by RANSAC

- **green circles**: dynamic rigid body transformations extracted by adaptive thresholds

- **green text**: L2 error of the rigid body transform in 2D and 3D / thresholds

### **Extracts dynamic rigid body transformations with strict policies.**

In my previous attempts at dynamic point extractions, I used a lenient policy that could mistakenly identify dynamic points as static points due to observation errors. While it achieved the intended goal, it also resulted in extracting a large number of static points. Therefore, I have decided to refine my approach once again.

To determine the dynamic points, I use a method similar to the previous one. I compare the two predictions separately in both 2D and 3D. This comparison is only done for points that consensus the rigid body transformation suggested by RANSAC. Here are the two predictions:

- The first prediction is performed by assuming that the point is dynamic. To get a refined transformation, I use SVD registration on the points at time `t-1` and `t`. Then, I apply the refined transformation to the points at time `t-1`.

- The second prediction assumes that the points are static. I apply the camera's transformation to the points at time `t-1`.

If the difference between these two predictions is larger than a certain threshold, I assume that the points are dynamic.

When **dealing with a large number of input points**, I simplify the SVD registration process by selecting **the k points with the smallest L2 values based on the rigid body transformation suggested by RANSAC**. This helps to limit the complexity of the registration.

To determine whether the points are dynamic or not, I calculate the difference for all points and divide it by the number of points. Then, I compare this value to the threshold in both 3D and 2D. Here are the details:

- **Threshold in 3D**: $\alpha + {{\beta}\over{N}}$
- **Threshold in 2D**: ${{\gamma}\over{\bar{z}}} + {{\delta}\over{N}}$
- $\alpha, \beta, \gamma, \delta$: constants
- $N$ : number of points
- $\bar{z}$: average depth of the points

When calculating the L2 difference in 3D, I also include a penalty in the z direction. This is to compensate for the depth inaccuracy of stereo cameras. For thresholds, **the second term helps to reduce uncertainty**, as using fewer points increases the probability of **overfitting based on the MLE**. More observations increase the probability of getting a distribution close to the truth by the law of large numbers.

For 2D comparisons, **the first term compensates** for the disadvantage of comparing differences in 2D for **points closer to the camera**. This is because the **image projection** of an object is **inversely proportional to its depth**. The second term serves the same purpose as in the 3D case. Similarly, for consistency, the error is evaluated for both viewpoints and the greater of the two is used..

I have imposed stricter constraints for dynamic point extraction, as the following reasons suggest:

- Instead of analyzing individual points, I grouped them to analyze L2, which is less susceptible to errors.
- Since I **assume a rigid transformation** and have **information at both time `t-1` and `t`**, the **accuracy is higher** compared to comparing the **observation at time `t`** to the prediction that applies the camera's transformation to the point at `t-1`.

## Object refinement using on-manifold optimization
![image](https://user-images.githubusercontent.com/72921481/229909167-247c43f8-30e4-4216-97ca-bb13d5b5419e.png)

![image](https://user-images.githubusercontent.com/72921481/226542049-9a5bf742-3b88-477b-9d34-cb8fe25bd12b.png)
![image](https://user-images.githubusercontent.com/72921481/226542088-362671ba-6cea-4c74-8524-359a64f4a948.png)

### **On-manifold optimize 3D points and transformations with respect to observations.**
To achieve accurate tracking information, I employ nonlinear optimization. This involves making slight adjustments to the 3D points and transformations at time t-1 to minimize the reprojection error. The initial values of the transforms are suggested by registration using SVD. The re-projection error is then evaluated for four views, resulting from assessing two time points across two viewpoints.

$$argmin \sum_{t=0}^{1} \sum_{i=0}^{1} \sum_{j=0}^{N-1} (\textbf{p} _ {i, t}-h(\textbf{T} _ i \hat{\textbf{T}} _ t \hat{\textbf{P}}))^T \Omega _ {i, t} (\textbf{p} _ {i, t}-h(\textbf{T} _ i \hat{\textbf{T}} _ t \hat{\textbf{P}}))$$
- $t$: time point
- $i$: camera index
- $j$: point index
- $N$: number of points
- $\textbf{p}_{i, t}$: 2D point
- $h()$: projection function
- $\textbf{T}_i$: transformation from left camera to right camera
- $\hat{\textbf{T}}_t$: transformation from time 0 to time t (parameter)
- $\hat{\textbf{P}}$: 3D point represented at left camera time 0 (parameter)
- $\Omega_{i, t}$: observation covariance

When extracting features, OpenVINS follows a process that involves extracting features from the left view at time `t-1` and propagating them to the right view using KLT. Then, the points of both views are propagated to time `t`. As a result, the points of the right view at time `t-1` and the left view at time `t` undergo two KLTs, while the points of the right view at time `t` undergo three KLTs. Due to this process, **the observations corresponding to the original intended left view at time `t-1` become increasingly inaccurate**. To compensate for this, I assigned a different observation covariance to the points based on the number of times KLT was performed.

While I could optimize the parameters subject to epipolar constraints, I decided to focus on the re-projection error to limit the complexity.

## Transformation association over time using the Hungarian algorithm
![image](https://user-images.githubusercontent.com/72921481/229910084-9b2f24bb-7de1-4a1f-b6ea-7a4a76992b75.png)

![image](https://user-images.githubusercontent.com/72921481/226544979-ec2e09cb-7e57-438f-b41e-e24b249e0904.png)
- **different colored circles**: 2D points assigned different transformations by color
- **arrow**: KLT tracking

### **Quickly associate transformations over time using the Hungarian algorithm.**

To track dynamic objects, I associate newly proposed transformations with past transformations. I do this by counting the **number of connected KLTs between past and present dynamic objects**. Based on this, I construct a **cost matrix** and use the Hungarian algorithm to find the associations. This has an `O(N^3)` complexity, where `N` is the number of proposed transformations.

To reduce outliers in real-world implementations, I assume that at least two KLT traces are required to be associated.


## Focusing on dynamic objects
![image](https://user-images.githubusercontent.com/72921481/229910724-f1857db4-d026-4559-af3c-eb59dc0b3dd0.png)

![Screenshot from 2023-04-02 22-38-56](https://user-images.githubusercontent.com/72921481/229356816-db28c492-6510-4e3d-8b57-ba76763bfbfe.png)

- **red-bordered green points**: dynamic points that are being tracked
- **normal green points**: additional proposed dynamic points around the tracked point
- **blue points**: points for SLAMs proposed outside the dynamic area.

### Increase tracking and localization accuracy by focusing more on dynamic objects.

To **isolate the dynamic points** from the SLAM system, they are tracked within an **independent data structure**. The grid occupied by the tracked points is classified as a **dynamic region**, where **additional dynamic points** are proposed, and all **SLAM points are removed**. This approach not only makes tracking more robust but also removes biased or dynamic points from the SLAM system. As a result, it reduces drift and enhances the robustness of the SLAM system.

In dynamic regions, **more dynamic points are intentionally extracted** than regular SLAM features.
