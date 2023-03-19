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

using namespace ov_type;
using namespace ov_msckf;


void get_hungarian_correspondences(const std::vector<std::vector<int>> &cost, std::vector<std::pair<size_t, size_t>> &correspondences); 

cv::Scalar randomColor();

int fact (const int n);

int nCr (const int n, const int r);

void serialize_graphs(std::vector<std::vector<size_t>> &graphs, const size_t num_pts, std::vector<std::vector<size_t>> &graphed_idcs);

void make_graphs(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_after, const size_t num_pts, const size_t num_nearest, std::vector<std::vector<size_t>> &graphs, cv::Mat &test_l, const std::shared_ptr<Vec> &calib, std::vector<cv::Point2d> &pts_now_viz, std::vector<cv::Point2d> &pts_before_viz);

void reject_static_pts(const std::shared_ptr<Vec> &calib, pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_before, pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_after, std::vector<size_t> &raw_idcs, const cv::Mat &raw_pts3d_before, const cv::Mat &raw_pts3d_after, const cv::Mat &raw_pts3d_pred, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, const std::vector<size_t> &raw_idcs_raw, cv::Mat &test_l, std::vector<cv::Point2d> &pts_now_viz, std::vector<cv::Point2d> &pts_before_viz, std::vector<cv::Point2d> &pts_now_viz1, std::vector<cv::Point2d> &pts_before_viz1);

void get_meas_and_pred(const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, cv::Mat &raw_pts3d_before, cv::Mat &raw_pts3d_after, cv::Mat &raw_pts3d_pred, Eigen::Matrix4d &T_C0BtoC0A);

bool get_sampled_idcs(std::vector<size_t> &sampled_idcs, std::vector<size_t> &rnd_base, const std::vector<size_t> &idcs, const size_t k, std::set<std::vector<size_t>> &used);

int get_max_iters(const double probability, const double outliers_ratio, const size_t k, const std::shared_ptr<ov_msckf::State> &state, const size_t num_idcs);

bool is_dynamic_tf(const size_t num_idcs, std::vector<uchar> &mask_out, const std::vector<size_t> &idcs, const pcl::PointCloud<pcl::PointXYZ> &pts_before, const pcl::PointCloud<pcl::PointXYZ> &pts_after, const std::vector<float> &full_size_L2_3D_vec, const int best_num_inliers, Eigen::Matrix4f &inliers_tf, const Eigen::Matrix4d &T_C0BtoC0A, const std::shared_ptr<ov_msckf::State> &state);

void ransac_tf(const pcl::PointCloud<pcl::PointXYZ> &pts_before, const pcl::PointCloud<pcl::PointXYZ> &pts_after,
                       const std::vector<size_t> &idcs, const float th_L2_3D_ransac, const double probability, bool &succeed, bool &dynamic, std::vector<uchar> &mask_out, Eigen::Matrix4f &inliers_tf, const Eigen::Matrix4d &T_C0BtoC0A, const std::shared_ptr<ov_msckf::State> &state);

void label_pts(std::vector<std::vector<std::vector<size_t>>> &labeled_all_idcs, std::vector<std::vector<char>> &labeled_all_states, std::vector<std::vector<Eigen::Matrix4f>> &labeled_all_tf, const std::vector<std::vector<size_t>> &graphed_idcs, const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_before, const pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_after, const Eigen::Matrix4d &T_C0BtoC0A, const std::shared_ptr<ov_msckf::State> &state);

void track_moving_objects(const ov_core::CameraData &message, const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, const std::vector<size_t> &raw_idcs_raw);