#include "Tracker.h"

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

void reject_static_pts(const std::shared_ptr<Vec> &calib, pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_before, pcl::PointCloud<pcl::PointXYZ>::Ptr &pts_after, std::vector<size_t> &raw_idcs, const cv::Mat &raw_pts3d_before, const cv::Mat &raw_pts3d_after, const cv::Mat &raw_pts3d_pred, const std::vector<std::vector<cv::Point2f>> &raw_pts_C0, const std::vector<std::vector<cv::Point2f>> &raw_pts_C1, const std::vector<size_t> &raw_idcs_raw, cv::Mat &test_l, std::vector<cv::Point2d> &pts_now_viz, std::vector<cv::Point2d> &pts_before_viz, std::vector<cv::Point2d> &pts_now_viz1, std::vector<cv::Point2d> &pts_before_viz1) {
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
pts_now_viz1.emplace_back(cv::Point2d((double)raw_pts_C1[1][i].x, (double)raw_pts_C1[1][i].y));
pts_before_viz1.emplace_back(cv::Point2d((double)raw_pts_C1[0][i].x, (double)raw_pts_C1[0][i].y));
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
printf("extrinsic before---------------------\n");
std::cout<<R_C0BtoC1B<<std::endl;
std::cout<<p_C0BinC1B<<std::endl;
printf("extrinsic after---------------------\n");
std::cout<<R_C0AtoC1A<<std::endl;
std::cout<<p_C0AinC1A<<std::endl;

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
std::vector<cv::Point2d> pts_now_viz1;
std::vector<cv::Point2d> pts_before_viz1;
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
  reject_static_pts(calib, pts_before, pts_after, raw_idcs, raw_pts3d_before, raw_pts3d_after, raw_pts3d_pred, raw_pts_C0, raw_pts_C1, raw_idcs_raw, test_l, pts_now_viz, pts_before_viz, pts_now_viz1, pts_before_viz1);
  
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

if(!labeled_raw_idcs.empty())
{
// Get current time in milliseconds
auto now = std::chrono::system_clock::now();
auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

// Convert milliseconds to time_t (seconds since epoch)
std::time_t time = static_cast<std::time_t>(millis / 1000);
std::string file_name0 = "/home/csh/test/imgs/" + std::to_string(time) + "_now0.png";
std::string file_name1 = "/home/csh/test/imgs/" + std::to_string(time) + "_now1.png";

cv::Mat test_l0;
cv::cvtColor(message.images.at(0), test_l0, cv::COLOR_GRAY2RGB);
cv::Mat test_r0;
cv::cvtColor(message.images.at(1), test_r0, cv::COLOR_GRAY2RGB);
// cv::imwrite(file_name0, test_l0);
// cv::imwrite(file_name1, test_r0);

printf("cam mat\n %f %f %f %f\n", calib->value()(0), calib->value()(1), calib->value()(2), calib->value()(3));

for(int i = 0; i<labeled_idcs.size(); ++i)
{
printf("tf------------\n");
std::cout<<labeled_tf[i]<<std::endl;
  auto &idcs = labeled_idcs[i];
printf("2d now 0 ------------------\n");
  for(int j = 0; j < idcs.size(); ++j)
  {
    int idx = idcs[j];
    printf("%f %f\n", pts_now_viz[idx].x, pts_now_viz[idx].y);
  }
printf("2d now 1 ------------------\n");
  for(int j = 0; j < idcs.size(); ++j)
  {
    int idx = idcs[j];
    printf("%f %f\n", pts_now_viz1[idx].x, pts_now_viz1[idx].y);
  }
printf("2d before 0 ------------------\n");
  for(int j = 0; j < idcs.size(); ++j)
  {
    int idx = idcs[j];
    printf("%f %f\n", pts_before_viz[idx].x, pts_before_viz[idx].y);
  }
printf("2d before 1 ------------------\n");
  for(int j = 0; j < idcs.size(); ++j)
  {
    int idx = idcs[j];
    printf("%f %f\n", pts_before_viz1[idx].x, pts_before_viz1[idx].y);
  }
printf("3d now ------------------\n");
  for(int j = 0; j < idcs.size(); ++j)
  {
    int idx = idcs[j];
    printf("%f %f %f\n", pts_after->at(idx).x, pts_after->at(idx).y, pts_after->at(idx).z);
  }
printf("3d before ------------------\n");
  for(int j = 0; j < idcs.size(); ++j)
  {
    int idx = idcs[j];
    printf("%f %f %f\n", pts_before->at(idx).x, pts_before->at(idx).y, pts_before->at(idx).z);
  }
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
printf("%d -> %d\n", correspondences[i].first, correspondences[i].second);
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