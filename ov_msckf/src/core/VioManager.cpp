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
#include <random>
#include <string>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void get_hungarian_correspondences(const std::vector<std::vector<int>> &cost, std::vector<std::pair<size_t, size_t>> &correspondences) {
  int n = cost.size();
  int m = cost[0].size();
  std::vector<int> u(n+1);
  std::vector<int> v(m+1);
  std::vector<int> p(m+1);
  std::vector<int> way(m+1);
  const int INF = std::numeric_limits<int>::max();

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<int> minv(m+1, INF);
    std::vector<bool> used(m+1, false);
    do {
        used[j0] = true;
        int i0 = p[j0];
        int delta = INF;
        int j1;
        for (int j = 1; j <= m; ++j) {
          if (used[j])
            continue;

          int cur = cost[i0-1][j-1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
        for (int j = 0; j <= m; ++j) {
          if (used[j]) {
            u[p[j]] += delta;
            v[j] -= delta;
          } else {
            minv[j] -= delta;
          }
        }
        j0 = j1;
      } while (p[j0] != 0);

      do {
        int j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0 != 0);
  }

  for (int j = 1; j <= m; ++j) {
    if (p[j] > 0) {
      correspondences.emplace_back(std::make_pair(size_t(p[j]-1), size_t(j-1)));
    }
  }
}

cv::Scalar randomColor() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    return cv::Scalar(dis(gen), dis(gen), dis(gen));
}

int fact (const int n) {
  if (n == 0)
    return 1;

  int res = 1;
  for (int i = 2; i <= n; ++i) {
    res = res * i;
  }
      
  return res;
}

int nCr (const int n, const int r) {
    return fact(n) / (fact(r) * fact(n - r));
}

void serialize_graphs(std::vector<std::vector<size_t>> &graphs, const size_t num_pts, std::vector<std::vector<size_t>> &graphed_idcs) {
  graphed_idcs = std::vector<std::vector<size_t>>(1);
  const size_t min_num_pts = 3;
  std::vector<bool> emplaced(num_pts, false);
  for (size_t i = 0; i<num_pts; ++i) {
    if (graphs[i].empty() || emplaced[i])
      continue;

    size_t num_pts = 0;
    std::stack<size_t> stack;
    emplaced[i] = true;
    stack.emplace(i);
    while (!stack.empty()) {
      size_t idx_now = stack.top();
      stack.pop();

      ++num_pts;
      graphed_idcs.back().emplace_back(idx_now);
      for (auto next_idx : graphs[idx_now]) {
        if (!emplaced[next_idx]) {
          emplaced[next_idx] = true;
          stack.emplace(next_idx);
        }
      }
    }

    if (num_pts < min_num_pts)
      graphed_idcs.back().clear();
    else
      graphed_idcs.emplace_back(std::vector<size_t>{});
  }
  if (!graphed_idcs.empty() && graphed_idcs.back().empty())
    graphed_idcs.pop_back();
}

void make_graphs(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_after, const size_t num_pts, const size_t num_nearest, std::vector<std::vector<size_t>> &graphs, cv::Mat &test_l, const std::shared_ptr<Vec> &calib, std::vector<cv::Point2d> &pts_now_viz, std::vector<cv::Point2d> &pts_before_viz) {
  graphs = std::vector<std::vector<size_t>>(num_pts);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  kd_tree.setInputCloud(pts_after);
  for (size_t i = 0; i<num_pts; ++i) {
    pcl::PointXYZ query_point = pts_after->at(i);

    // Search for the k nearest neighbors to the query point
    std::vector<int> idcs(num_nearest+1);
    std::vector<float> dists(num_nearest+1);
    kd_tree.nearestKSearch(query_point, num_nearest+1, idcs, dists);
auto from = cv::Point2d(calib->value()(0)*pts_now_viz[i].x+calib->value()(2), calib->value()(1)*pts_now_viz[i].y+calib->value()(3));
    for (size_t j = 0; j<= num_nearest; ++j) {
      // Graph only points that are close in distance and not query points
      if (dists[j] < 2.0 && (size_t)idcs[j] != i) {
        graphs[i].emplace_back((size_t)idcs[j]);
        graphs[idcs[j]].emplace_back(i);
auto from_before = cv::Point2d(calib->value()(0)*pts_before_viz[i].x+calib->value()(2), calib->value()(1)*pts_before_viz[i].y+calib->value()(3));
cv::arrowedLine(test_l, from_before, from, cv::Scalar(255,0,0), 2);
cv::putText(test_l, std::to_string(i), from, 1, 2, cv::Scalar(0,0,255), 2);
auto to = cv::Point2d(calib->value()(0)*pts_now_viz[idcs[j]].x+calib->value()(2), calib->value()(1)*pts_now_viz[idcs[j]].y+calib->value()(3));
auto to_before = cv::Point2d(calib->value()(0)*pts_before_viz[idcs[j]].x+calib->value()(2), calib->value()(1)*pts_before_viz[idcs[j]].y+calib->value()(3));
cv::arrowedLine(test_l, to_before, to, cv::Scalar(255,0,0), 2);
cv::putText(test_l, std::to_string(idcs[j]), to, 1, 2, cv::Scalar(0,0,255), 2);
cv::line(test_l, from, to, cv::Scalar(0, 0, 255), 1);
cv::circle(test_l, to, 3, cv::Scalar(255,255,255), 3);
      }
    }
  }
}

void reject_static_pts(const std::shared_ptr<Vec> &calib, pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_before, pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_after, std::vector<size_t> &raw_idcs, const cv::Mat &raw_pts3d_before, const cv::Mat &raw_pts3d_after, const cv::Mat &raw_pts3d_pred, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, const std::vector<size_t> &raw_idcs_raw, cv::Mat &test_l, std::vector<cv::Point2d> &pts_now_viz, std::vector<cv::Point2d> &pts_before_viz) {
  pts_after = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pts_before = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  size_t num_pts = raw_pts3d_before.cols;
  for(size_t i = 0; i<num_pts; ++i) {
    // Predicted points in a Euclidean frame
    double x_pred = raw_pts3d_pred.at<double>(0, i)/raw_pts3d_pred.at<double>(3, i);
    double y_pred = raw_pts3d_pred.at<double>(1, i)/raw_pts3d_pred.at<double>(3, i);
    double z_pred = raw_pts3d_pred.at<double>(2, i)/raw_pts3d_pred.at<double>(3, i);

    // Observed points in a Euclidean frame
    double x_after = raw_pts3d_after.at<double>(0, i)/raw_pts3d_after.at<double>(3, i);
    double y_after = raw_pts3d_after.at<double>(1, i)/raw_pts3d_after.at<double>(3, i);
    double z_after = raw_pts3d_after.at<double>(2, i)/raw_pts3d_after.at<double>(3, i);

    // L2 in Euclidean with z-direction penalty
    double weighted_L2_3d = sqrt(pow(x_pred - x_after, 2) + pow(y_pred - y_after, 2) + (1/1.5) * pow(z_pred - z_after, 2));

    // reprojection error
    double L2_2d = sqrt(pow(x_pred/z_pred - x_after/x_pred, 2) + pow(y_pred/z_pred - y_after/y_pred, 2));
    // Insert dynamic points into a point cloud
    if (z_pred < 100 && z_after < 100 && weighted_L2_3d < 100 && (L2_2d > 5/calib->value()(0) || weighted_L2_3d > 0.1)) {
      double x_before = raw_pts3d_before.at<double>(0, i)/raw_pts3d_before.at<double>(3, i);
      double y_before = raw_pts3d_before.at<double>(1, i)/raw_pts3d_before.at<double>(3, i);
      double z_before = raw_pts3d_before.at<double>(2, i)/raw_pts3d_before.at<double>(3, i);
      pts_after->emplace_back(pcl::PointXYZ{(float)x_after, (float)y_after, (float)z_after});
      pts_before->emplace_back(pcl::PointXYZ{(float)x_before, (float)y_before, (float)z_before});
      raw_idcs.emplace_back(raw_idcs_raw[i]);
pts_now_viz.emplace_back(cv::Point2d((double)raw_pts_C0[1][i].x, (double)raw_pts_C0[1][i].y));
pts_before_viz.emplace_back(cv::Point2d((double)raw_pts_C0[0][i].x, (double)raw_pts_C0[0][i].y));
    }
  }
}

void get_meas_and_pred(const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, cv::Mat &raw_pts3d_before, cv::Mat &raw_pts3d_after, cv::Mat &raw_pts3d_pred, Eigen::Matrix4d &T_C0BtoC0A) {
  auto R_GtoIA = Eigen::Matrix3d(state->_imu->Rot());
  auto p_IAinG = Eigen::Vector3d(state->_imu->pos());
  auto R_ItoC0 = Eigen::Matrix3d(state->_calib_IMUtoCAM[0]->Rot());
  auto R_ItoC1 = Eigen::Matrix3d(state->_calib_IMUtoCAM[1]->Rot());
  auto p_IinC0 = Eigen::Vector3d(state->_calib_IMUtoCAM[0]->pos());
  auto p_IinC1 = Eigen::Vector3d(state->_calib_IMUtoCAM[1]->pos());
  auto R_GtoIB = state->_R_GtoIB;
  auto p_IBinG = state->_p_IBinG;

  // criteria point : {global}
  Eigen::Matrix3d R_GtoC0B = R_ItoC0 * R_GtoIB;
  Eigen::Matrix3d R_GtoC0A = R_ItoC0 * R_GtoIA;
  Eigen::Matrix3d R_GtoC1B = R_ItoC1 * R_GtoIB;
  Eigen::Matrix3d R_GtoC1A = R_ItoC1 * R_GtoIA;
  Eigen::Vector3d p_GinC0B = p_IinC0 - R_GtoC0B * p_IBinG;
  Eigen::Vector3d p_GinC0A = p_IinC0 - R_GtoC0A * p_IAinG;
  Eigen::Vector3d p_GinC1B = p_IinC1 - R_GtoC1B * p_IBinG;
  Eigen::Vector3d p_GinC1A = p_IinC1 - R_GtoC1A * p_IAinG;

  // criteria point : {cam0}
  Eigen::Matrix3d R_C0BtoCOB = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_C0AtoC0A = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_C0BtoC1B = R_GtoC1B * R_GtoC0B.transpose();
  Eigen::Matrix3d R_C0AtoC1A = R_GtoC1A * R_GtoC0A.transpose();
  Eigen::Vector3d p_C0BinC0B = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_C0AinC0A = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_C0BinC1B = p_GinC1B - R_C0BtoC1B * p_GinC0B;
  Eigen::Vector3d p_C0AinC1A = p_GinC1A - R_C0AtoC1A * p_GinC0A;

  //Since I've been passed normalized points, the projection matrix is an extrinsic
  Eigen::Matrix<double, 3, 4> P_C0B_temp;
  P_C0B_temp << R_C0BtoCOB, p_C0BinC0B;
  Eigen::Matrix<double, 3, 4> P_C0A_temp;
  P_C0A_temp << R_C0AtoC0A, p_C0AinC0A;
  Eigen::Matrix<double, 3, 4> P_C1B_temp;
  P_C1B_temp << R_C0BtoC1B, p_C0BinC1B;
  Eigen::Matrix<double, 3, 4> P_C1A_temp;
  P_C1A_temp << R_C0AtoC1A, p_C0AinC1A;
  cv::Mat P_C0B;
  cv::Mat P_C0A;
  cv::Mat P_C1B;
  cv::Mat P_C1A;
  cv::eigen2cv(P_C0B_temp, P_C0B);
  cv::eigen2cv(P_C0A_temp, P_C0A);
  cv::eigen2cv(P_C1B_temp, P_C1B);
  cv::eigen2cv(P_C1A_temp, P_C1A);
  cv::triangulatePoints(P_C0B, P_C1B, raw_pts_C0[0], raw_pts_C1[0], raw_pts3d_before);
  cv::triangulatePoints(P_C0A, P_C1A, raw_pts_C0[1], raw_pts_C1[1], raw_pts3d_after);

  // Compute a prediction of the points for motion, assuming the received points are static
  Eigen::Matrix3d R_C0BtoC0A = R_GtoC0A * R_GtoC0B.transpose();
  Eigen::Vector3d p_C0BinC0A = p_GinC0A - R_C0BtoC0A * p_GinC0B;
  Eigen::Matrix<double, 3, 4> P_C0BtoC0A;
  P_C0BtoC0A << R_C0BtoC0A, p_C0BinC0A;
  Eigen::Matrix<double, 1, 4> T_bottom;
  T_bottom << 0.0, 0.0, 0.0, 1.0;
  T_C0BtoC0A << P_C0BtoC0A, T_bottom;
  cv::Mat T_C0BtoC0A_cv;
  cv::eigen2cv(T_C0BtoC0A, T_C0BtoC0A_cv);
  raw_pts3d_before.convertTo(raw_pts3d_before, CV_64F);
  raw_pts3d_after.convertTo(raw_pts3d_after, CV_64F);
  raw_pts3d_pred = T_C0BtoC0A_cv * raw_pts3d_before;
}

bool get_sampled_idcs(std::vector<size_t> &sampled_idcs, std::vector<size_t> &rnd_base, const std::vector<size_t> &idcs, const size_t k, std::set<std::vector<size_t>> &used) {
  std::random_device rd;
  std::mt19937 gen(rd());
  sampled_idcs.resize(k);
  
  std::shuffle(rnd_base.begin(), rnd_base.end(), gen);
  for (size_t k_i = 0; k_i<k; ++k_i) {
    sampled_idcs[k_i] = idcs[rnd_base[k_i]];
  }
  std::sort(sampled_idcs.begin(), sampled_idcs.end());

  if (used.find(sampled_idcs) == used.end()) {
    used.emplace(sampled_idcs);
    return true;
  }
  else {
    return false;
  }
}

int get_max_iters(const double probability, const double outliers_ratio, const size_t k, const std::shared_ptr<ov_msckf::State> &state, const size_t num_idcs) {
  auto it = state->_iter_table.find(num_idcs);
  if (it == state->_iter_table.end()) {
    int max_iters = log(1 - probability) / log(1 - pow(1 - 0.3, k));
    if (num_idcs < 10) {
      max_iters = std::min(max_iters, nCr(num_idcs, k));
    }

    state->_iter_table.emplace(num_idcs, max_iters);
    return max_iters;
  }
  
  return it->second;
}

bool is_dynamic_tf(const size_t num_idcs, std::vector<uchar> &mask_out, const std::vector<size_t> &idcs, const pcl::PointCloud<pcl::PointXYZ> &pts_before, const pcl::PointCloud<pcl::PointXYZ> &pts_after, const std::vector<float> &full_size_L2_3D_vec, const int best_num_inliers, Eigen::Matrix4f &inliers_tf, const Eigen::Matrix4d &T_C0BtoC0A, const std::shared_ptr<ov_msckf::State> &state) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_before_temp(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_after_temp(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<float> L2_3D_vec_temp;
  for (size_t full_idx = 0; full_idx<num_idcs; ++full_idx) {
    if (mask_out[full_idx] == 0)
      continue;

    size_t pts_idx = idcs[full_idx];
    inliers_before_temp->emplace_back(pts_before[pts_idx]);
    inliers_after_temp->emplace_back(pts_after[pts_idx]);
    L2_3D_vec_temp.emplace_back(full_size_L2_3D_vec[full_idx]);
  }

  std::vector<float> sorting_idx(L2_3D_vec_temp);
  std::iota(sorting_idx.begin(), sorting_idx.end(), 0);
  std::sort(sorting_idx.begin(), sorting_idx.end(), 
            [&](float i0, float i1) {return L2_3D_vec_temp[i0] < L2_3D_vec_temp[i1];});

  int max_elements = std::min(best_num_inliers, 10);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_before(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_after(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<float> L2_3D_vec;
  inliers_before->resize(max_elements);
  inliers_after->resize(max_elements);
  L2_3D_vec.resize(max_elements);
  for (size_t idx_to = 0; idx_to<(size_t)max_elements; ++idx_to)
  {
    size_t idx_from = sorting_idx[idx_to];
    inliers_before->at(idx_to) = inliers_before_temp->at(idx_from);
    inliers_after->at(idx_to) = inliers_after_temp->at(idx_from);
    L2_3D_vec[idx_to] = L2_3D_vec_temp[idx_from];
  }

  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
  svd.estimateRigidTransformation(*inliers_before, *inliers_after, inliers_tf);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pred_pts(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*inliers_before, *inliers_pred_pts, inliers_tf);

  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pred_ego(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f T_C0BtoC0A_f = T_C0BtoC0A.cast<float>();
  pcl::transformPointCloud(*inliers_before, *inliers_pred_ego, T_C0BtoC0A_f);

  double weighted_L2_3D = 0.0;
  double L2_2D = 0.0;
  double z_mean = 0.0;
  for (size_t i = 0; i<(size_t)max_elements; ++i) {
    Eigen::Vector3f weighted_diff_3D = inliers_pred_pts->at(i).getVector3fMap() - inliers_pred_ego->at(i).getVector3fMap();
    weighted_diff_3D.block<1, 1>(2, 0) *= (1/1.5);
    weighted_L2_3D += (double)weighted_diff_3D.norm();
    double u_pred_ego = (double)inliers_pred_ego->at(i).x/(double)inliers_pred_ego->at(i).z;
    double v_pred_ego = (double)inliers_pred_ego->at(i).y/(double)inliers_pred_ego->at(i).z;
    double u_pred_pts = (double)inliers_pred_pts->at(i).x/(double)inliers_pred_pts->at(i).z;
    double v_pred_pts = (double)inliers_pred_pts->at(i).y/(double)inliers_pred_pts->at(i).z;
    auto pt_pred_ego = cv::Point2d(u_pred_ego, v_pred_ego);
    auto pt_pred_pts = cv::Point2d(u_pred_pts, v_pred_pts);
    cv::Point2d diff_2D = pt_pred_ego-pt_pred_pts;
    L2_2D += cv::norm(diff_2D);
    z_mean += inliers_pred_pts->at(i).z;
  }
  auto fx = state->_cam_intrinsics[0]->value()(0);
  weighted_L2_3D /= max_elements;
  L2_2D *= fx/max_elements;
  z_mean /= max_elements;
  double th_2D = 20/z_mean + 15/max_elements;
  double th_3D = 0.15 + 0.1/max_elements;

printf("3d : %f / %f\n", weighted_L2_3D, th_3D);
printf("2d : %f / %f\n", L2_2D, th_2D);
printf("dynamic? %d\n", z_mean > 0.0 && weighted_L2_3D > th_3D && L2_2D > th_2D);
  if(z_mean > 0.0 && weighted_L2_3D > th_3D && L2_2D > th_2D) {
    return true;
  }

  return false;
}

void ransac_tf(const pcl::PointCloud<pcl::PointXYZ> &pts_before, const pcl::PointCloud<pcl::PointXYZ> &pts_after,
                       const std::vector<size_t> &idcs, const float th_L2_3D_ransac, const double probability, bool &succeed, bool &dynamic, std::vector<uchar> &mask_out, Eigen::Matrix4f &inliers_tf, const Eigen::Matrix4d &T_C0BtoC0A, const std::shared_ptr<ov_msckf::State> &state) {
printf("input : ");
for(int i = 0;i<idcs.size(); ++i)
printf("%d ", idcs[i]);
printf("\n");
  // Initialize the RANSAC parameters
  // Number of points to draw for each iteration
  const size_t k = 3; 
  size_t num_idcs = idcs.size();
  int max_iters = get_max_iters(probability, 0.3, k, state, num_idcs);
printf("max_iters %d \n", max_iters);
  int best_num_inliers = -1;
  std::vector<uchar> inliers_mask;
  std::vector<float> full_size_L2_3D_vec;
  std::vector<size_t> sampled_idcs;
  std::vector<size_t> rnd_base(idcs.size());
  std::iota(rnd_base.begin(), rnd_base.end(), 0);
  std::set<std::vector<size_t>> used;

  // Iterate for a maximum number of iterations
  int fail_cnt = 0;
  for (int iter = 0; iter < max_iters && fail_cnt<50; iter++) {
    bool sample_succeed = get_sampled_idcs(sampled_idcs, rnd_base, idcs, k, used);
    if (!sample_succeed) {
        --iter;
        ++fail_cnt;
        continue;
    }

    // Estimate the transformation using the k random points
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_pts_before(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_pts_after(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i<k; ++i) {
      sampled_pts_before->emplace_back(pts_before[sampled_idcs[i]]);
      sampled_pts_after->emplace_back(pts_after[sampled_idcs[i]]);
    }
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    Eigen::Matrix4f sampled_pts_tf;
    svd.estimateRigidTransformation(*sampled_pts_before, *sampled_pts_after, sampled_pts_tf);

    // Evaluate the transformation using the available points
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_pts_pred(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(pts_before, *sampled_pts_pred, sampled_pts_tf);

    int num_inliers = 0;
    inliers_mask = std::vector<uchar>(idcs.size(), 0);
    full_size_L2_3D_vec = std::vector<float>(idcs.size(), std::numeric_limits<float>::max());
    for (size_t i = 0; i<num_idcs; ++i) {
      float sampled_L2_3D = (sampled_pts_pred->at(idcs[i]).getVector3fMap() - pts_after[idcs[i]].getVector3fMap()).norm();
      if (sampled_L2_3D < th_L2_3D_ransac) {
        ++num_inliers;
        inliers_mask[i] = 1;
        full_size_L2_3D_vec[i] = sampled_L2_3D;
      }
    }

    // Update the best transformation if the current transformation has more inliers
    if (num_inliers > best_num_inliers) {
      best_num_inliers = num_inliers;
      mask_out = inliers_mask;
    }
  }
  
  if (best_num_inliers >= k) {
    succeed = true;
  }
  else {
    succeed = false;
  }

printf("res (%d ea) :", best_num_inliers);
for(int i = 0; i<mask_out.size(); ++i)
{
  if(mask_out[i] == 1)
    printf("%d ", idcs[i]);
}
std::cout<<std::endl;

printf("succeed? %d\n", succeed);

  if (succeed) {
    dynamic = false;
    dynamic = is_dynamic_tf(num_idcs, mask_out, idcs, pts_before, pts_after, full_size_L2_3D_vec, best_num_inliers, inliers_tf, T_C0BtoC0A, state);
  }
}

void label_pts(std::vector<std::vector<std::vector<size_t>>> &labeled_all_idcs, std::vector<std::vector<char>> &labeled_all_states, std::vector<std::vector<Eigen::Matrix4f>> &labeled_all_tf, const std::vector<std::vector<size_t>> &graphed_idcs, const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_before, const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_after, const Eigen::Matrix4d &T_C0BtoC0A, const std::shared_ptr<ov_msckf::State> &state) {
  labeled_all_idcs = std::vector<std::vector<std::vector<size_t>>>(graphed_idcs.size());
  labeled_all_states = std::vector<std::vector<char>>(graphed_idcs.size());
  labeled_all_tf = std::vector<std::vector<Eigen::Matrix4f>>(graphed_idcs.size());
  size_t num_graph = graphed_idcs.size();
  for (size_t graph_idx = 0; graph_idx<num_graph; ++graph_idx) {
    bool succeed = true;
    bool dinamic = false;
    std::vector<size_t> idcs_now = graphed_idcs[graph_idx];
    std::vector<uchar> mask(graphed_idcs[graph_idx].size(), 0);
    Eigen::Matrix4f tf_now;
    ransac_tf(*pts_before, *pts_after, idcs_now, 0.1, 0.99, succeed, dinamic, mask, tf_now, T_C0BtoC0A, state);
    while (succeed) {
      std::vector<size_t> idcs_next;
      std::vector<size_t> inlier_idcs;
      for (size_t mask_idx = 0; mask_idx < mask.size(); ++mask_idx) {
        if (mask[mask_idx] == 1) {
          inlier_idcs.emplace_back(idcs_now[mask_idx]);
        }
        else {
          idcs_next.emplace_back(idcs_now[mask_idx]);
        }
      }

      labeled_all_idcs[graph_idx].emplace_back(inlier_idcs);
      labeled_all_tf[graph_idx].emplace_back(tf_now);
      if (dinamic) {
        labeled_all_states[graph_idx].emplace_back(1);
      }
      else {
        labeled_all_states[graph_idx].emplace_back(2);
      }

printf("next suggestion : ");
for(int i = 0; i<idcs_next.size(); ++i)
printf("%d ", idcs_next[i]);
printf("\n");

      if (idcs_next.size() < 3) {
        labeled_all_idcs[graph_idx].emplace_back(idcs_next);
        labeled_all_tf[graph_idx].emplace_back(Eigen::Matrix4f());
        labeled_all_states[graph_idx].emplace_back(0);
        break;
      }

      idcs_now = idcs_next;
      mask = std::vector<uchar> (idcs_next.size(), 0);
      Eigen::Matrix4f tf_next;
      ransac_tf(*pts_before, *pts_after, idcs_next, 0.1, 0.99, succeed, dinamic, mask, tf_next, T_C0BtoC0A, state);

      if (!succeed) {
        labeled_all_idcs[graph_idx].emplace_back(idcs_next);
        labeled_all_tf[graph_idx].emplace_back(Eigen::Matrix4f());
        labeled_all_states[graph_idx].emplace_back(0);
      } 
    }
  }
}

void track_moving_objects(const ov_core::CameraData &message, const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, const std::vector<size_t> &raw_idcs_raw) {
cv::Mat test_l;
cv::cvtColor(message.images.at(0), test_l, cv::COLOR_GRAY2RGB);
std::vector<cv::Point2d> pts_now_viz;
std::vector<cv::Point2d> pts_before_viz;
  if (raw_pts_C0[0].empty()) {
cv::imshow("test", test_l);
cv::waitKey(1);
    return;
  }

  cv::Mat raw_pts3d_before;
  cv::Mat raw_pts3d_after;
  cv::Mat raw_pts3d_pred;
  Eigen::Matrix4d T_C0BtoC0A;
  get_meas_and_pred(state, raw_pts_C0, raw_pts_C1, raw_pts3d_before, raw_pts3d_after, raw_pts3d_pred, T_C0BtoC0A);

  // Receive an instruction that changes every time
  std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(0);
  // reject static pts, get only dynamic pts
  pcl::PointCloud<pcl::PointXYZ>::Ptr pts_after;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pts_before;
  std::vector<size_t> raw_idcs;
  reject_static_pts(calib, pts_before, pts_after, raw_idcs, raw_pts3d_before, raw_pts3d_after, raw_pts3d_pred, raw_pts_C0, raw_pts_C1, raw_idcs_raw, test_l, pts_now_viz, pts_before_viz);
  
  // Get k nearest points in the neighbourhood
  const size_t num_nearest = 3;
  size_t num_pts = pts_after->size();
  if (num_pts > num_nearest) {
    std::vector<std::vector<size_t>> graphs;
    make_graphs(pts_after, num_pts, num_nearest, graphs, test_l, calib, pts_now_viz, pts_before_viz);

    // DFS to bring all elements of a graphs together and reject graphs with fewer nodes
    std::vector<std::vector<size_t>> graphed_idcs;
    serialize_graphs(graphs, num_pts, graphed_idcs);

for(int i = 0; i<graphed_idcs.size(); ++i) {
for(int j = 0; j <graphed_idcs[i].size(); ++j) {
printf("%d ", graphed_idcs[i][j]);
}
std::cout<<std::endl;
}

    std::vector<std::vector<std::vector<size_t>>> labeled_all_idcs;
    std::vector<std::vector<char>> labeled_all_states;
    std::vector<std::vector<Eigen::Matrix4f>> labeled_all_tf;
    label_pts(labeled_all_idcs, labeled_all_states, labeled_all_tf, graphed_idcs, pts_before, pts_after, T_C0BtoC0A, state);

    std::vector<std::vector<size_t>> labeled_idcs;
    std::vector<Eigen::Matrix4f> labeled_tf;
    std::vector<std::vector<size_t>> labeled_raw_idcs;
    for (size_t graph_idx = 0; graph_idx<labeled_all_idcs.size(); ++graph_idx) {
      for (size_t label_idx = 0; label_idx<labeled_all_idcs[graph_idx].size(); ++label_idx) {
        if(labeled_all_states[graph_idx][label_idx] != 1)
          continue;

        std::vector<size_t> partial_raw_idcs;
        for (size_t pt_idx = 0; pt_idx<labeled_all_idcs[graph_idx][label_idx].size(); ++pt_idx) {
          size_t idx = labeled_all_idcs[graph_idx][label_idx][pt_idx];
          partial_raw_idcs.emplace_back(raw_idcs[idx]);
        }
        labeled_raw_idcs.emplace_back(partial_raw_idcs);
        labeled_idcs.emplace_back(labeled_all_idcs[graph_idx][label_idx]);
        labeled_tf.emplace_back(labeled_all_tf[graph_idx][label_idx]);
      }
    }

    
    size_t num_labels_now = labeled_raw_idcs.size();
    std::unordered_map<size_t, size_t> raw_idcs_table_now;
    for (size_t label_idx = 0; label_idx<num_labels_now; ++label_idx) {
      for (size_t pt_idx = 0; pt_idx<labeled_raw_idcs[label_idx].size(); ++pt_idx) {
        size_t raw_idx = labeled_raw_idcs[label_idx][pt_idx];
        raw_idcs_table_now.emplace(raw_idx, label_idx);
      }
    } 
    if (num_labels_now > 0 && state->_num_labels_before > 0)
    {
      size_t mat_size = std::max(state->_num_labels_before, num_labels_now);
      std::vector<std::vector<int>> cost_matrix(mat_size, std::vector<int>(mat_size, 0));
      int min_val = 0;
      for (size_t label_idx_now = 0; label_idx_now<num_labels_now; ++label_idx_now) {
printf("input : ");
for (size_t pt_idx = 0; pt_idx<labeled_raw_idcs[label_idx_now].size(); ++pt_idx) {
printf("%d(%d) ", labeled_idcs[label_idx_now][pt_idx], labeled_raw_idcs[label_idx_now][pt_idx]);
}
std::cout<<std::endl;

        for (size_t pt_idx = 0; pt_idx < labeled_raw_idcs[label_idx_now].size(); ++pt_idx) {
          size_t raw_idx_now = labeled_raw_idcs[label_idx_now][pt_idx];
          auto it_before = state->_raw_idcs_table_before.find(raw_idx_now);
          if(it_before == state->_raw_idcs_table_before.end())
            continue;

          size_t label_idx_before = it_before->second;
          --cost_matrix[label_idx_now][label_idx_before];
          min_val = std::min(min_val, cost_matrix[label_idx_now][label_idx_before]);
        }
      }

printf("table\n");
printf("     ");
for(int i = 0; i<cost_matrix[0].size(); ++i)
printf("%d ", i);
std::cout<<std::endl;
for(int i = 0; i<cost_matrix.size(); ++i)
{
  printf("%d : ", i);
  for(int j = 0; j<cost_matrix[i].size(); ++j)
  {
    printf("%d ", cost_matrix[i][j]);
  }
  std::cout<<std::endl;
}

    for(size_t i = 0; i<cost_matrix.size(); ++i)
    {
      for(int j = 0; j<cost_matrix[i].size(); ++j)
      {
        cost_matrix[i][j] -= min_val;
      }
      std::cout<<std::endl;
    }

printf("cost\n");
printf("     ");
for(int i = 0; i<cost_matrix[0].size(); ++i)
printf("%d ", i);
std::cout<<std::endl;
for(int i = 0; i<cost_matrix.size(); ++i)
{
printf("%d : ", i);
for(int j = 0; j<cost_matrix[i].size(); ++j)
{
printf("%d ", cost_matrix[i][j]);
}
std::cout<<std::endl;
}
std::cout<<std::endl;

    std::vector<std::pair<size_t, size_t>> correspondences;
    get_hungarian_correspondences(cost_matrix, correspondences);

printf("correspondences\n");
for(int i = 0; i<correspondences.size(); ++i)
{
if(cost_matrix[correspondences[i].first][correspondences[i].second] >= -min_val)
  continue;
printf("%d %d\n", correspondences[i].first, correspondences[i].second);
}
    }
    state->_raw_idcs_table_before = raw_idcs_table_now;
    state->_num_labels_before = num_labels_now;
    

    

printf("labels\n");
for(int i = 0; i<labeled_all_idcs.size(); ++i)
{
printf("graphs %d :\n", i);
for(int j = 0; j<labeled_all_idcs[i].size(); ++j)
{
auto label_color = randomColor();
if(labeled_all_states[i][j] != 1)
label_color = cv::Scalar(255,255,255);
printf("{");
for(size_t k = 0; k<labeled_all_idcs[i][j].size(); ++k)
{
printf("%d ", labeled_all_idcs[i][j][k]);
int idx = labeled_all_idcs[i][j][k];
auto q = cv::Point2d(calib->value()(0)*pts_now_viz[idx].x+calib->value()(2), calib->value()(1)*pts_now_viz[idx].y+calib->value()(3));
cv::circle(test_l, q, 3, label_color, 3);
if(labeled_all_states[i][j] == 1)
{
cv::circle(test_l, q, 13, cv::Scalar(0,255,0), 2);
std::string s = std::to_string(raw_idcs[idx]);
cv::putText(test_l, s, q, 1, 1, cv::Scalar(0, 255, 0));
}
}
printf("}(%d) ", labeled_all_states[i][j]);
}
printf("\n");
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

  auto R_IBtoC0 = Eigen::Matrix3d(state->_calib_IMUtoCAM[0]->Rot());
  auto R_IBtoC1 = Eigen::Matrix3d(state->_calib_IMUtoCAM[1]->Rot());
  auto p_IBinC0 = Eigen::Vector3d(state->_calib_IMUtoCAM[0]->pos());
  auto p_IBinC1 = Eigen::Vector3d(state->_calib_IMUtoCAM[1]->pos());
  std::vector<std::vector<cv::Point2f>> tracked_raw_pts_C0(2);
  std::vector<std::vector<cv::Point2f>> tracked_raw_pts_C1(2);
  std::vector<size_t> raw_idcs;
  // Perform our feature tracking!
  if (params.use_stereo) {
    auto R_C0toC1 = R_IBtoC1 * R_IBtoC0.transpose();
    auto p_C0inC1 = - R_C0toC1 * p_IBinC0 + p_IBinC1;
    trackFEATS->feed_new_camera(message, tracked_raw_pts_C0, tracked_raw_pts_C1, raw_idcs, R_C0toC1, p_C0inC1);
  }
  else {
    trackFEATS->feed_new_camera(message, tracked_raw_pts_C0, tracked_raw_pts_C1, raw_idcs);
  }

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message, tracked_raw_pts_C0, tracked_raw_pts_C1, raw_idcs);
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

      track_moving_objects(message, state, tracked_raw_pts_C0, tracked_raw_pts_C1, raw_idcs);
      state->_R_GtoIB = Eigen::Matrix3d(state->_imu->Rot());
      state->_p_IBinG = Eigen::Vector3d(state->_imu->pos());
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
  track_moving_objects(message, state, tracked_raw_pts_C0, tracked_raw_pts_C1, raw_idcs);
  state->_R_GtoIB = Eigen::Matrix3d(state->_imu->Rot());
  state->_p_IBinG = Eigen::Vector3d(state->_imu->pos());
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
