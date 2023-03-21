#pragma once

#include "VioManager.h"
#include "state/State.h"

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <random>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace ov_msckf;



class ObjectTracker {
public:
  void track(const ov_core::CameraData &message, const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, const std::vector<size_t> &raw_idcs_raw);

private:
  class ReprojectionErrorCostFunction;
  const int NUM_NEAREST = 3;
  const int NUM_MAX_ITER = 10;
  const int MIN_NUM_TF = 3;
  const int MAX_NUM_SVD = 10;
  const uchar DYNAMIC = 1;
  const uchar STATIC = 2;
  const uchar IS_INLIER = 1;
  const uchar IS_OUTLIER = 0;
  const float TH_L2_3D_RANSAC = 0.1;
  const double PROB_RANSAC = 0.99;
  const double OUTLIERS_RATIO = 0.3;
  const double Z_PENALTY = 1/1.5;
  const double ALPHA = 0.15;
  const double BETA = 0.1;
  const double DELTA = 20;
  const double GAMMA = 15;

  double fx0;
  double fy0;
  double cx0;
  double cy0;
  double fx1;
  double fy1;
  double cx1;
  double cy1;

  cv::Mat test_l;
  
  // pts
  std::vector<Eigen::Vector2d> p2ds_C0_now;
  
  std::vector<Eigen::Vector2d> p2ds_C0_prev;
  
  std::vector<Eigen::Vector2d> p2ds_C1_now;
  
  std::vector<Eigen::Vector2d> p2ds_C1_prev;

  pcl::PointCloud<pcl::PointXYZ>::Ptr p3ds_prev;
 
  pcl::PointCloud<pcl::PointXYZ>::Ptr p3ds_now;
  
  // tracker idcs
  std::vector<size_t> raw_idcs;

  // extrinsics
  Eigen::Matrix3d R_C0toC1_now;
  
  Eigen::Vector3d p_C0inC1_now;

  Eigen::Matrix4d T_C0_prev_to_now;
  
  Eigen::Matrix3d _R_GtoI_prev;
  
  Eigen::Vector3d _p_IinG_prev;

  // global  
  std::unordered_map<int, int> iter_table;
  
  std::unordered_map<size_t, size_t> _raw_idcs_table_prev;
  
  size_t _num_labels_prev;

  void init(const std::shared_ptr<ov_msckf::State> &state);

  bool pass_ransac(const std::vector<size_t> &idcs, const float th_L2_3d, const double prob, std::vector<uchar> &mask_out, std::vector<float> &L2_3d_vec, int &best_num_consensus);

  void get_hungarian_pairs(const std::vector<std::vector<int>> &cost, std::vector<std::pair<size_t, size_t>> &pairs);

  cv::Scalar randomColor();

  int fact (const int n);

  int nCr (const int n, const int r);
  
  void rm_outliers(const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C0, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C1, const std::vector<size_t> &all_raw_idcs);

  void serialize_graphs(std::vector<std::vector<size_t>> &graphs, std::vector<std::vector<size_t>> &graphed_idcs);

  void divide_graphs(std::vector<std::vector<size_t>> &graphed_idcs, std::vector<std::vector<size_t>> &labeled_idcs, std::vector<Eigen::Matrix4f> &labeled_tf, std::vector<std::vector<size_t>> &labeled_raw_idcs);

  void make_graphs(std::vector<std::vector<size_t>> &graphs);

  void reject_static_p3ds(const cv::Mat &all_p3ds_prev, const cv::Mat &all_p3ds_now, const cv::Mat &all_p3ds_pred, const std::vector<std::vector<cv::Point2f>> &all_p2ds_C0, const std::vector<std::vector<cv::Point2f>> &all_p2ds_C1, const std::vector<size_t> &all_raw_idcs);

  void get_meas_and_pred(const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &all_p2ds_C0, const std::vector<std::vector<cv::Point2f>> &all_p2ds_C1, cv::Mat &all_p3ds_prev, cv::Mat &all_p3ds_now, cv::Mat &all_p3ds_pred);

  bool get_sampled_idcs(std::vector<size_t> &sampled_idcs, std::vector<size_t> &rnd_base, const std::vector<size_t> &idcs, const size_t k, std::set<std::vector<size_t>> &used);

  int get_max_iters(const double probability, const double outliers_ratio, const size_t k, const size_t num_idcs);

  bool pass_svd(std::vector<uchar> &mask_out, const std::vector<size_t> &idcs, const std::vector<float> &full_size_L2_3d_vec, const int best_num_inliers, Eigen::Matrix4f &inliers_tf);

  void test_dynamic(const std::vector<size_t> &idcs, bool &succeed, bool &dynamic, std::vector<uchar> &mask_out, Eigen::Matrix4f &inliers_tf);

  void label_p3ds(std::vector<std::vector<std::vector<size_t>>> &labeled_all_idcs, std::vector<std::vector<uchar>> &labeled_all_states, std::vector<std::vector<Eigen::Matrix4f>> &labeled_all_tf, const std::vector<std::vector<size_t>> &graphed_idcs);
};
