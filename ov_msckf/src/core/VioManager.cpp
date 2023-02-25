/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2022 Patrick Geneva
 * Copyright (C) 2018-2022 Guoquan Huang
 * Copyright (C) 2018-2022 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;



void serialize_dynamic_pts_graph(std::vector<std::vector<size_t>> &dynamic_pts_graph, const size_t num_dynamic_pts, std::vector<std::vector<size_t>> &dynamic_pts_indices_group) {
  size_t min_num_pts_ransac = 3;
  std::vector<bool> emplaced(num_dynamic_pts, false);
  for (size_t i = 0; i<num_dynamic_pts; ++i) {
    if (dynamic_pts_graph[i].empty() || emplaced[i])
      continue;

    size_t cnt = 0;
    std::stack<size_t> s;
    emplaced[i] = true;
    s.emplace(i);
    while (!s.empty()) {
      size_t idx_now = s.top();
      s.pop();

      ++cnt;
      dynamic_pts_indices_group.back().emplace_back(idx_now);
      for (auto next_idx : dynamic_pts_graph[idx_now]) {
        if (!emplaced[next_idx]) {
          emplaced[next_idx] = true;
          s.emplace(next_idx);
        }
      }
    }

    if (cnt < min_num_pts_ransac)
      dynamic_pts_indices_group.back().clear();
    else
      dynamic_pts_indices_group.emplace_back(std::vector<size_t>{});
  }
  if (!dynamic_pts_indices_group.empty() && dynamic_pts_indices_group.back().empty())
    dynamic_pts_indices_group.pop_back();
}

void construct_dynamic_pts_graph(const pcl::PointCloud<pcl::PointXYZ>::Ptr &dynamic_pts_after, const size_t num_dynamic_pts, const size_t num_nearest, std::vector<std::vector<size_t>> &dynamic_pts_graph, cv::Mat &test_l, const std::shared_ptr<Vec> &calib, std::vector<cv::Point2d> &dynamic_pts_viz) {
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  kd_tree.setInputCloud(dynamic_pts_after);
  for (size_t i = 0; i<num_dynamic_pts; ++i) {
    pcl::PointXYZ query_point = dynamic_pts_after->at(i);

    // Search for the k nearest neighbors to the query point
    std::vector<int> indices(num_nearest+1);
    std::vector<float> distances(num_nearest+1);
    kd_tree.nearestKSearch(query_point, num_nearest+1, indices, distances);
auto q = cv::Point2d(calib->value()(0)*dynamic_pts_viz[i].x+calib->value()(2), calib->value()(1)*dynamic_pts_viz[i].y+calib->value()(3));
    for (size_t j = 0; j<= num_nearest; ++j) {
      // Graph only points that are close in distance and not query points
      if (distances[j] < 2.0 && (size_t)indices[j] != i) {
        dynamic_pts_graph[i].emplace_back((size_t)indices[j]);
        dynamic_pts_graph[indices[j]].emplace_back(i);
printf("%d -> %d\n", i, indices[j]);
printf("%d -> %d\n", indices[j], i);
auto a = cv::Point2d(calib->value()(0)*dynamic_pts_viz[indices[j]].x+calib->value()(2), calib->value()(1)*dynamic_pts_viz[indices[j]].y+calib->value()(3));
cv::line(test_l, q, a, cv::Scalar(0, 0, 255), 1);
      }
    }
  }
}

void reject_static_pts(const std::shared_ptr<Vec> &calib, const pcl::PointCloud<pcl::PointXYZ>::Ptr &dynamic_pts_before, const pcl::PointCloud<pcl::PointXYZ>::Ptr &dynamic_pts_after, const cv::Mat &pts3d_before, const cv::Mat &pts3d_after, const cv::Mat &pts3d_predict, const std::vector<std::vector<cv::Point2f>> &ransac_denied_pts_C0, const std::vector<std::vector<cv::Point2f>> &ransac_denied_pts_C1, cv::Mat &test_l, std::vector<cv::Point2d> &dynamic_pts_viz) {
  size_t num_pts = pts3d_before.cols;
  for(size_t i = 0; i<num_pts; ++i) {
auto lb = cv::Point2d(calib->value()(0)*(double)ransac_denied_pts_C0[0][i].x+calib->value()(2), calib->value()(1)*(double)ransac_denied_pts_C0[0][i].y+calib->value()(3));
auto la = cv::Point2d(calib->value()(0)*(double)ransac_denied_pts_C0[1][i].x+calib->value()(2), calib->value()(1)*(double)ransac_denied_pts_C0[1][i].y+calib->value()(3));
    // Predicted points in a Euclidean frame
    double x_predict = pts3d_predict.at<double>(0, i)/pts3d_predict.at<double>(3, i);
    double y_predict = pts3d_predict.at<double>(1, i)/pts3d_predict.at<double>(3, i);
    double z_predict = pts3d_predict.at<double>(2, i)/pts3d_predict.at<double>(3, i);

    // Observed points in a Euclidean frame
    double x_after = pts3d_after.at<double>(0, i)/pts3d_after.at<double>(3, i);
    double y_after = pts3d_after.at<double>(1, i)/pts3d_after.at<double>(3, i);
    double z_after = pts3d_after.at<double>(2, i)/pts3d_after.at<double>(3, i);

    // L2 in Euclidean with z-direction penalty
    double weighted_L2_3d = sqrt(pow(x_predict - x_after, 2) + pow(y_predict - y_after, 2) + (1/1.5) * pow(z_predict - z_after, 2));

    // reprojection error
    double L2_2d = sqrt(pow(x_predict/z_predict - (double)ransac_denied_pts_C0[1][i].x, 2) + pow(y_predict/z_predict - (double)ransac_denied_pts_C0[1][i].y, 2));
#include <string>
    // Insert dynamic points into a point cloud
    if (z_predict < 50 && z_after < 50 && 0.5 < weighted_L2_3d && weighted_L2_3d < 100 && 15/calib->value()(0) < L2_2d) {
      double x_before = pts3d_before.at<double>(0, i)/pts3d_before.at<double>(3, i);
      double y_before = pts3d_before.at<double>(1, i)/pts3d_before.at<double>(3, i);
      double z_before = pts3d_before.at<double>(2, i)/pts3d_before.at<double>(3, i);
      dynamic_pts_after->emplace_back(pcl::PointXYZ{(float)x_after, (float)y_after, (float)z_after});
      dynamic_pts_before->emplace_back(pcl::PointXYZ{(float)x_before, (float)y_before, (float)z_before});
dynamic_pts_viz.emplace_back(cv::Point2d((double)ransac_denied_pts_C0[1][i].x, (double)ransac_denied_pts_C0[1][i].y));
cv::circle(test_l, la, 3, cv::Scalar(255,255,255), 3);
cv::arrowedLine(test_l, lb, la, cv::Scalar(255,0,0), 2);
cv::putText(test_l, std::to_string(dynamic_pts_viz.size()-1), la, 1, 2, cv::Scalar(255,0,0), 2);
    }
  }
}

void get_measurement_and_prediction(const std::shared_ptr<ov_msckf::State> &state, const Eigen::Matrix3d &R_IBtoC0, const Eigen::Matrix3d &R_GtoIB, const Eigen::Matrix3d &R_IBtoC1, Eigen::Vector3d &p_IBinC0, Eigen::Vector3d &p_IBinG, Eigen::Vector3d &p_IBinC1, const std::vector<std::vector<cv::Point2f>> &ransac_denied_pts_C0, const std::vector<std::vector<cv::Point2f>> &ransac_denied_pts_C1, cv::Mat &pts3d_before, cv::Mat &pts3d_after, cv::Mat &pts3d_predict) {
    auto R_GtoIA = Eigen::Matrix3d(state->_imu->Rot());
    auto p_IAinG = Eigen::Vector3d(state->_imu->pos());
    auto R_IAtoC0 = Eigen::Matrix3d(state->_calib_IMUtoCAM[0]->Rot());
    auto R_IAtoC1 = Eigen::Matrix3d(state->_calib_IMUtoCAM[1]->Rot());
    auto p_IAinC0 = Eigen::Vector3d(state->_calib_IMUtoCAM[0]->pos());
    auto p_IAinC1 = Eigen::Vector3d(state->_calib_IMUtoCAM[1]->pos());

    // criteria point : {global}
    Eigen::Matrix3d R_GtoC0B = R_IBtoC0 * R_GtoIB;
    Eigen::Matrix3d R_GtoC0A = R_IAtoC0 * R_GtoIA;
    Eigen::Matrix3d R_GtoC1B = R_IBtoC1 * R_GtoIB;
    Eigen::Matrix3d R_GtoC1A = R_IAtoC1 * R_GtoIA;
    Eigen::Vector3d p_GinC0B = p_IBinC0 - R_GtoC0B * p_IBinG;
    Eigen::Vector3d p_GinC0A = p_IAinC0 - R_GtoC0A * p_IAinG;
    Eigen::Vector3d p_GinC1B = p_IBinC1 - R_GtoC1B * p_IBinG;
    Eigen::Vector3d p_GinC1A = p_IAinC1 - R_GtoC1A * p_IAinG;

    // criteria point : {cam0} when (t-1)
    Eigen::Matrix3d R_C0BtoCOB = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_C0BtoC0A = R_GtoC0A * R_GtoC0B.transpose();
    Eigen::Matrix3d R_C0BtoC1B = R_GtoC1B * R_GtoC0B.transpose();
    Eigen::Matrix3d R_C0BtoC1A = R_GtoC1A * R_GtoC0B.transpose();
    Eigen::Vector3d p_C0BinC0B = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_C0BinC0A = p_GinC0A - R_C0BtoC0A * p_GinC0B;
    Eigen::Vector3d p_C0BinC1B = p_GinC1B - R_C0BtoC1B * p_GinC0B;
    Eigen::Vector3d p_C0BinC1A = p_GinC1A - R_C0BtoC1A * p_GinC0B;

    //Since I've been passed normalized points, the projection matrix is an extrinsic
    Eigen::Matrix<double, 3, 4> P_C0B_temp;
    P_C0B_temp << R_C0BtoCOB, p_C0BinC0B;
    Eigen::Matrix<double, 3, 4> P_C0A_temp;
    P_C0A_temp << R_C0BtoC0A, p_C0BinC0A;
    Eigen::Matrix<double, 3, 4> P_C1B_temp;
    P_C1B_temp << R_C0BtoC1B, p_C0BinC1B;
    Eigen::Matrix<double, 3, 4> P_C1A_temp;
    P_C1A_temp << R_C0BtoC1A, p_C0BinC1A;
    cv::Mat P_C0B;
    cv::Mat P_C0A;
    cv::Mat P_C1B;
    cv::Mat P_C1A;
    cv::eigen2cv(P_C0B_temp, P_C0B);
    cv::eigen2cv(P_C0A_temp, P_C0A);
    cv::eigen2cv(P_C1B_temp, P_C1B);
    cv::eigen2cv(P_C1A_temp, P_C1A);
    cv::triangulatePoints(P_C0B, P_C1B, ransac_denied_pts_C0[0], ransac_denied_pts_C1[0], pts3d_before);
    cv::triangulatePoints(P_C0A, P_C1A, ransac_denied_pts_C0[1], ransac_denied_pts_C1[1], pts3d_after);

    // Compute a prediction of the points for motion, assuming the received points are static
    cv::Mat T_BtoA;
    cv::Mat T_bottom = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 1.0);
    cv::vconcat(P_C0A, T_bottom, T_BtoA);
    pts3d_before.convertTo(pts3d_before, CV_64F);
    pts3d_after.convertTo(pts3d_after, CV_64F);
    pts3d_predict = T_BtoA * pts3d_before;
}

Eigen::Matrix4f ransac_transformation(const pcl::PointCloud<pcl::PointXYZ> &pts_before, const pcl::PointCloud<pcl::PointXYZ> &pts_after,
                       const std::vector<size_t> &indices, float threshold, float probability) {

  // Initialize the RANSAC parameters
  const int k = 3; // Number of points to draw for each iteration
  const int max_iterations = log(1 - probability) / log(1 - pow(1 - 0.3, k));
printf("max_iterations %d :\n", max_iterations);
  int best_num_inliers = 0;
  Eigen::Matrix4f best_transformation;

  // Create a random number generator
  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_int_distribution<int> distribution(0, indices.size() - 1);

  std::vector<int> combination_mask(indices.size(), 0);
  for (int i = 0; i < indices.size()-k; ++i) {
    combination_mask[i] = 0;
  }
  for (int i = indices.size()-k; i < indices.size(); ++i) {
    combination_mask[i] = 1;
  }

  int iteration = 0;
  // Iterate for a maximum number of iterations
  do {

    // Draw k random points from the indices
    // std::vector<int> sample_indices(k);
    // for (int i = 0; i < k; ++i) {
    //     int index = distribution(gen);
    //     sample_indices[i] = indices[index];
    // }
printf("suggestion idcs \n");
    // Estimate the transformation using the k random points
    pcl::PointCloud<pcl::PointXYZ>::Ptr sample_before(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr sample_after(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < indices.size(); ++i) {
      if(combination_mask[i] == 0)
        continue;

printf("%d ", indices[i]);
      sample_before->emplace_back(pts_before[indices[i]]);
      sample_after->emplace_back(pts_after[indices[i]]);
    }
printf("\n");
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    Eigen::Matrix4f transformation;
    svd.estimateRigidTransformation(*sample_before, *sample_after, transformation);

    // Evaluate the transformation using the available points
    int num_inliers = 0;
    for (int i = 0; i < indices.size(); ++i) {
      pcl::PointXYZ transformed_point;
      transformed_point.getVector3fMap() = transformation.block<3, 3>(0, 0) * pts_before[indices[i]].getVector3fMap() + transformation.block<3, 1>(0, 3);
      float L2 = (transformed_point.getVector3fMap() - pts_after[indices[i]].getVector3fMap()).norm();
      if (L2 < threshold) {
        ++num_inliers;
      }
    }

    // Update the best transformation if the current transformation has more inliers
    if (num_inliers > best_num_inliers) {
      best_num_inliers = num_inliers;
      best_transformation = transformation;
printf("renew \n");
printf("best_num_inliers : \n", best_num_inliers);
for(auto i : combination_mask)
printf("%d ", i);
printf("\n");
    }

    ++iteration;
  } while(iteration != max_iterations && std::next_permutation(combination_mask.begin(), combination_mask.end()));

printf("res (%d ea)\n", best_num_inliers);
std::cout<<best_transformation<<std::endl;
  // Return the best transformation
  return best_transformation;
}

void track_moving_objects(const ov_core::CameraData &message, const std::shared_ptr<ov_msckf::State> &state, const Eigen::Matrix3d &R_IBtoC0, const Eigen::Matrix3d &R_GtoIB, const Eigen::Matrix3d &R_IBtoC1, Eigen::Vector3d &p_IBinC0, Eigen::Vector3d &p_IBinG, Eigen::Vector3d &p_IBinC1, const std::vector<std::vector<cv::Point2f>> &ransac_denied_pts_C0, const std::vector<std::vector<cv::Point2f>> &ransac_denied_pts_C1) {
cv::Mat test_l;
cv::cvtColor(message.images.at(0), test_l, cv::COLOR_GRAY2RGB);
std::vector<cv::Point2d> dynamic_pts_viz;
  if (ransac_denied_pts_C0[0].empty()) {
cv::imshow("test", test_l);
cv::waitKey(1);
    return;
  }

  cv::Mat pts3d_before;
  cv::Mat pts3d_after;
  cv::Mat pts3d_predict;
  get_measurement_and_prediction(state, R_IBtoC0, R_GtoIB, R_IBtoC1, p_IBinC0, p_IBinG, p_IBinC1, ransac_denied_pts_C0, ransac_denied_pts_C1, pts3d_before, pts3d_after, pts3d_predict);


  // Receive an instruction that changes every time
  std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(0);
  // reject static pts, get only dynamic pts
  pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_pts_after(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr dynamic_pts_before(new pcl::PointCloud<pcl::PointXYZ>);
  reject_static_pts(calib, dynamic_pts_before, dynamic_pts_after, pts3d_before, pts3d_after, pts3d_predict, ransac_denied_pts_C0, ransac_denied_pts_C1, test_l, dynamic_pts_viz);
  
  // Get k nearest points in the neighbourhood
  size_t num_nearest = 3;
  size_t num_dynamic_pts = dynamic_pts_after->size();
  if (num_dynamic_pts > num_nearest) {
    std::vector<std::vector<size_t>> dynamic_pts_graph(num_dynamic_pts);
    construct_dynamic_pts_graph(dynamic_pts_after, num_dynamic_pts, num_nearest, dynamic_pts_graph, test_l, calib, dynamic_pts_viz);

for(int i = 0; i<dynamic_pts_graph.size(); ++i) {
printf("%d : ", i);
for(auto j : dynamic_pts_graph[i]) {
printf("%d ", j);
}
std::cout<<std::endl;
}

    // DFS to bring all elements of a graph together and reject graphs with fewer nodes
    std::vector<std::vector<size_t>> dynamic_pts_indices_group(1);
    serialize_dynamic_pts_graph(dynamic_pts_graph, num_dynamic_pts, dynamic_pts_indices_group);


for(int i = 0; i<dynamic_pts_indices_group.size(); ++i) {
for(int j = 0; j <dynamic_pts_indices_group[i].size(); ++j) {
printf("%d ", dynamic_pts_indices_group[i][j]);
}
std::cout<<std::endl;
}
for(int i = 0; i<dynamic_pts_indices_group.size(); ++i) {
for(int j = 0; j < dynamic_pts_indices_group[i].size()-1; ++j) {
for(int k = j+1; k<dynamic_pts_indices_group[i].size(); ++k) {
auto q = cv::Point2d(calib->value()(0)*dynamic_pts_viz[dynamic_pts_indices_group[i][j]].x+calib->value()(2), calib->value()(1)*dynamic_pts_viz[dynamic_pts_indices_group[i][j]].y+calib->value()(3));
auto a = cv::Point2d(calib->value()(0)*dynamic_pts_viz[dynamic_pts_indices_group[i][k]].x+calib->value()(2), calib->value()(1)*dynamic_pts_viz[dynamic_pts_indices_group[i][k]].y+calib->value()(3));
cv::line(test_l, q, a, cv::Scalar(0, 255, 0), 1);
}
}
}
    int num_dynamic_groups = dynamic_pts_indices_group.size();
    for (const auto &indices : dynamic_pts_indices_group) {
      ransac_transformation(*dynamic_pts_before, *dynamic_pts_after, indices, 0.2, 0.99);
    }
  }
cv::imshow("test", test_l);
cv::waitKey(1);
}

VioManager::VioManager(VioManagerOptions &params_) : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }

  // Feature initializer for active tracks
  active_tracks_initializer = std::make_shared<FeatureInitializer>(params.featinit_options);
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state->margtimestep();
  if (oldest_time > state->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }


  auto R_GtoIB = Eigen::Matrix3d(state->_imu->Rot());
  auto p_IBinG = Eigen::Vector3d(state->_imu->pos());
  auto R_IBtoC0 = Eigen::Matrix3d(state->_calib_IMUtoCAM[0]->Rot());
  auto R_IBtoC1 = Eigen::Matrix3d(state->_calib_IMUtoCAM[1]->Rot());
  auto p_IBinC0 = Eigen::Vector3d(state->_calib_IMUtoCAM[0]->pos());
  auto p_IBinC1 = Eigen::Vector3d(state->_calib_IMUtoCAM[1]->pos());
  std::vector<std::vector<cv::Point2f>> ransac_denied_pts_C0(2);
  std::vector<std::vector<cv::Point2f>> ransac_denied_pts_C1(2);
  // Perform our feature tracking!
  if (params.use_stereo) {
    auto R_C0toC1 = R_IBtoC1 * R_IBtoC0.transpose();
    auto p_C0inC1 = - R_C0toC1 * p_IBinC0 + p_IBinC1;
    trackFEATS->feed_new_camera(message, ransac_denied_pts_C0, ransac_denied_pts_C1, R_C0toC1, p_C0inC1);
  }
  else {
    trackFEATS->feed_new_camera(message, ransac_denied_pts_C0, ransac_denied_pts_C1);
  }

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message, ransac_denied_pts_C0, ransac_denied_pts_C1);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
printf("mot from zupt\n");
      track_moving_objects(message, state, R_IBtoC0, R_GtoIB, R_IBtoC1, p_IBinC0, p_IBinG, p_IBinC1, ransac_denied_pts_C0, ransac_denied_pts_C1);
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(message);

  // select dynamic pts
printf("mot from normal\n");
  track_moving_objects(message, state, R_IBtoC0, R_GtoIB, R_IBtoC1, p_IBinC0, p_IBinG, p_IBinC1, ransac_denied_pts_C0, ransac_denied_pts_C1);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    propagator->propagate_and_clone(state, message.timestamp);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  updaterMSCKF->update(state, featsup_MSCKF);
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }
}
