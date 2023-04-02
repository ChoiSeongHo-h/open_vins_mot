#include "ObjectTracker.h"


class ObjectTracker::ReprojectionErrorCostFunction {
 public:
  ReprojectionErrorCostFunction(const Eigen::Vector2d &p2ds_C0_prev,
                                const Eigen::Vector2d &p2ds_C1_prev,
                                const Eigen::Vector2d &p2ds_C0_now,
                                const Eigen::Vector2d &p2ds_C1_now,
                                const Eigen::Quaterniond &q_C0toC1,
                                const Eigen::Vector3d &p_C0inC1_now,
                                const double &fx,
                                const double &fy)
      : p2ds_C0_prev_(p2ds_C0_prev),
        p2ds_C1_prev_(p2ds_C1_prev),
        p2ds_C0_now_(p2ds_C0_now),
        p2ds_C1_now_(p2ds_C1_now),
        q_C0toC1_(q_C0toC1),
        p_C0inC1_now_(p_C0inC1_now),
        fx_(fx),
        fy_(fy) {}

  template <typename T>
  bool operator()(const T* const p3d_i, const T* const q_para_i, const T* const p_para_i, T* residuals) const {
    Eigen::Matrix<T, 2, 1> p2ds_C0_prev = p2ds_C0_prev_.cast<T>();
    Eigen::Matrix<T, 2, 1> p2ds_C1_prev = p2ds_C1_prev_.cast<T>();
    Eigen::Matrix<T, 2, 1> p2ds_C0_now = p2ds_C0_now_.cast<T>();
    Eigen::Matrix<T, 2, 1> p2ds_C1_now = p2ds_C1_now_.cast<T>();
    Eigen::Quaternion<T> q_C0toC1 = q_C0toC1_.cast<T>();
    Eigen::Matrix<T, 3, 1> p_C0inC1_now = p_C0inC1_now_.cast<T>();
    T fx = T(fx_);
    T fy = T(fy_);
      
    Eigen::Quaternion<T> q_para(q_para_i[0], q_para_i[1], q_para_i[2], q_para_i[3]);
    Eigen::Matrix<T, 3, 1> p_para(p_para_i[0], p_para_i[1], p_para_i[2]);

    Eigen::Matrix<T, 3, 1> p3d(p3d_i[0], p3d_i[1], p3d_i[2]);
    T u_C0_prev = p3d(0)/p3d(2);
    T v_C0_prev = p3d(1)/p3d(2);
    residuals[0] = (u_C0_prev - T(p2ds_C0_prev(0))) / T(2.0/fx);
    residuals[1] = (v_C0_prev - T(p2ds_C0_prev(1))) / T(2.0/fy);

    Eigen::Matrix<T, 3, 1> p3d_C1_prev;
    p3d_C1_prev = q_C0toC1 * p3d + p_C0inC1_now;
    T u_C1_prev = p3d_C1_prev(0)/p3d_C1_prev(2);
    T v_C1_prev = p3d_C1_prev(1)/p3d_C1_prev(2);
    residuals[2] = (u_C1_prev - T(p2ds_C1_prev(0))) / T(4.0/fx);
    residuals[3] = (v_C1_prev - T(p2ds_C1_prev(1))) / T(4.0/fy);

    Eigen::Matrix<T, 3, 1> p3d_C0_now;
    p3d_C0_now = q_para * p3d + p_para;
    T u_C0_now = p3d_C0_now(0)/p3d_C0_now(2);
    T v_C0_now = p3d_C0_now(1)/p3d_C0_now(2);
    residuals[4] = (u_C0_now - T(p2ds_C0_now(0))) / T(4.0/fx);
    residuals[5] = (v_C0_now - T(p2ds_C0_now(1))) / T(4.0/fy);

    Eigen::Matrix<T, 3, 1> p3d_C1_now;
    p3d_C1_now = q_C0toC1 * p3d_C0_now + p_C0inC1_now;
    T u_C1_now = p3d_C1_now(0)/p3d_C1_now(2);
    T v_C1_now = p3d_C1_now(1)/p3d_C1_now(2);
    residuals[6] = (u_C1_now - T(p2ds_C1_now(0))) / T(6.0/fx);
    residuals[7] = (v_C1_now - T(p2ds_C1_now(1))) / T(6.0/fy);

    return true;
  }

 private:
  const Eigen::Vector2d &p2ds_C0_prev_;
  const Eigen::Vector2d &p2ds_C1_prev_;
  const Eigen::Vector2d &p2ds_C0_now_;
  const Eigen::Vector2d &p2ds_C1_now_;
  const Eigen::Quaterniond &q_C0toC1_;
  const Eigen::Vector3d &p_C0inC1_now_;
  const double &fx_;
  const double &fy_;
};

void ObjectTracker::get_hungarian_pairs(const std::vector<std::vector<int>> &cost, std::vector<std::pair<size_t, size_t>> &pairs) {
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
        int j1 = 0;
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
      pairs.emplace_back(std::make_pair(size_t(p[j]-1), size_t(j-1)));
    }
  }
}

cv::Scalar ObjectTracker::get_random_color() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    return cv::Scalar(dis(gen), dis(gen), dis(gen));
}

int ObjectTracker::fact (const int n) {
  if (n == 0)
    return 1;

  int res = 1;
  for (int i = 2; i <= n; ++i) {
    res = res * i;
  }
      
  return res;
}

int ObjectTracker::nCr (const int n, const int r) {
  return fact(n) / (fact(r) * fact(n - r));
}

void ObjectTracker::serialize_graphs(std::vector<std::vector<size_t>> &graphs, std::vector<std::vector<size_t>> &graphed_idcs) {
  auto num_p3ds = p3ds_now->size();
  graphed_idcs = std::vector<std::vector<size_t>>(1);
  const size_t min_p3ds = 3;
  std::vector<bool> emplaced(num_p3ds, false);
  for (size_t i = 0; i<num_p3ds; ++i) {
    if (graphs[i].empty() || emplaced[i])
      continue;

    size_t num_p3ds = 0;
    std::stack<size_t> stack;
    emplaced[i] = true;
    stack.emplace(i);
    while (!stack.empty()) {
      size_t idx_now = stack.top();
      stack.pop();

      ++num_p3ds;
      graphed_idcs.back().emplace_back(idx_now);
      for (auto next_idx : graphs[idx_now]) {
        if (!emplaced[next_idx]) {
          emplaced[next_idx] = true;
          stack.emplace(next_idx);
        }
      }
    }

    if (num_p3ds < min_p3ds)
      graphed_idcs.back().clear();
    else
      graphed_idcs.emplace_back(std::vector<size_t>{});
  }
  if (!graphed_idcs.empty() && graphed_idcs.back().empty())
    graphed_idcs.pop_back();
}

void ObjectTracker::make_graphs(std::vector<std::vector<size_t>> &graphs) {
  auto num_p3ds = p3ds_now->size();
  graphs = std::vector<std::vector<size_t>>(num_p3ds);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  kd_tree.setInputCloud(p3ds_now);
  for (size_t i = 0; i<num_p3ds; ++i) {
    pcl::PointXYZ query_point = p3ds_now->at(i);

    // Search for the k nearest neighbors to the query point
    std::vector<int> idcs(NUM_NEAREST+1);
    std::vector<float> dists(NUM_NEAREST+1);
    kd_tree.nearestKSearch(query_point, NUM_NEAREST+1, idcs, dists);
auto from = cv::Point2d(fx0*p2ds_C0_now[i].x()+cx0, fy0*p2ds_C0_now[i].y()+cy0);
    for (size_t j = 0; j<= NUM_NEAREST; ++j) {
      // Graph only points that are close in distance and not query points
      if (dists[j] < 2.0 && (size_t)idcs[j] != i) {
        graphs[i].emplace_back((size_t)idcs[j]);
        graphs[idcs[j]].emplace_back(i);
auto from_prev = cv::Point2d(fx0*p2ds_C0_prev[i].x()+cx0, fy0*p2ds_C0_prev[i].y()+cy0);
cv::arrowedLine(test_l, from_prev, from, cv::Scalar(255,0,0), 2);
// cv::putText(test_l, std::to_string(i), from, 1, 2, cv::Scalar(0,0,255), 2);
auto to = cv::Point2d(fx0*p2ds_C0_now[idcs[j]].x()+cx0, fy0*p2ds_C0_now[idcs[j]].y()+cy0);
auto to_prev = cv::Point2d(fx0*p2ds_C0_prev[idcs[j]].x()+cx0, fy0*p2ds_C0_prev[idcs[j]].y()+cy0);
cv::arrowedLine(test_l, to_prev, to, cv::Scalar(255,0,0), 2);
// cv::putText(test_l, std::to_string(idcs[j]), to, 1, 2, cv::Scalar(0,0,255), 2);
cv::line(test_l, from, to, cv::Scalar(0, 0, 255), 1);
cv::circle(test_l, to, 3, cv::Scalar(255,255,255), 3);
      }
    }
  }
}

void ObjectTracker::reject_static_p3ds(const cv::Mat &all_p3ds_prev, const cv::Mat &all_p3ds_now, const cv::Mat &all_p3ds_pred, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C0, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C1, const std::vector<size_t> &all_raw_idcs) {
  size_t num_p3ds = all_p3ds_prev.cols;
  for(size_t i = 0; i<num_p3ds; ++i) {
    // Predicted points in a Euclidean frame
    double x_pred = all_p3ds_pred.at<double>(0, i)/all_p3ds_pred.at<double>(3, i);
    double y_pred = all_p3ds_pred.at<double>(1, i)/all_p3ds_pred.at<double>(3, i);
    double z_pred = all_p3ds_pred.at<double>(2, i)/all_p3ds_pred.at<double>(3, i);

    // Observed points in a Euclidean frame
    double x_now = all_p3ds_now.at<double>(0, i)/all_p3ds_now.at<double>(3, i);
    double y_now = all_p3ds_now.at<double>(1, i)/all_p3ds_now.at<double>(3, i);
    double z_now = all_p3ds_now.at<double>(2, i)/all_p3ds_now.at<double>(3, i);

    // L2 in Euclidean with z-direction penalty
    double weighted_L2_3d = sqrt(pow(x_pred - x_now, 2) + pow(y_pred - y_now, 2) + (1/1.5) * pow(z_pred - z_now, 2));

    // reprojection error
    double L2_2d = sqrt(pow(x_pred/z_pred - x_now/x_pred, 2) + pow(y_pred/z_pred - y_now/y_pred, 2));
    // Insert is_dynamic points into a point cloud
    auto &all_p2ds_C0_prev = all_p2ds_set_C0[0];
    auto &all_p2ds_C1_prev = all_p2ds_set_C1[0];
    auto &all_p2ds_C0_now = all_p2ds_set_C0[1];
    auto &all_p2ds_C1_now = all_p2ds_set_C1[1];
    if (z_pred < 100 && z_now < 100 && weighted_L2_3d < 100 && (L2_2d > 5/fx0 || weighted_L2_3d > 0.1)) {
      double x_prev = all_p3ds_prev.at<double>(0, i)/all_p3ds_prev.at<double>(3, i);
      double y_prev = all_p3ds_prev.at<double>(1, i)/all_p3ds_prev.at<double>(3, i);
      double z_prev = all_p3ds_prev.at<double>(2, i)/all_p3ds_prev.at<double>(3, i);
      p3ds_now->emplace_back(pcl::PointXYZ{(float)x_now, (float)y_now, (float)z_now});
      p3ds_prev->emplace_back(pcl::PointXYZ{(float)x_prev, (float)y_prev, (float)z_prev});
      raw_idcs.emplace_back(all_raw_idcs[i]);
      Eigen::Vector2d p2d_C0_now;
      Eigen::Vector2d p2d_C0_prev;
      Eigen::Vector2d p2d_C1_now;
      Eigen::Vector2d p2d_C1_prev;
      p2d_C0_now << (double)all_p2ds_C0_now[i].x, (double)all_p2ds_C0_now[i].y;
      p2d_C0_prev << (double)all_p2ds_C0_prev[i].x, (double)all_p2ds_C0_prev[i].y;
      p2d_C1_now << (double)all_p2ds_C1_now[i].x, (double)all_p2ds_C1_now[i].y;
      p2d_C1_prev << (double)all_p2ds_C1_prev[i].x, (double)all_p2ds_C1_prev[i].y;
      p2ds_C0_now.emplace_back(p2d_C0_now);
      p2ds_C0_prev.emplace_back(p2d_C0_prev);
      p2ds_C1_now.emplace_back(p2d_C1_now);
      p2ds_C1_prev.emplace_back(p2d_C1_prev);
    }
  }
}

void ObjectTracker::get_meas_and_pred(const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C0, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C1, cv::Mat &all_p3ds_prev, cv::Mat &all_p3ds_now, cv::Mat &all_p3ds_pred) {
  auto R_GtoI_now = Eigen::Matrix3d(state->_imu->Rot());
  auto p_IinG_now = Eigen::Vector3d(state->_imu->pos());
  auto R_ItoC0 = Eigen::Matrix3d(state->_calib_IMUtoCAM[0]->Rot());
  auto R_ItoC1 = Eigen::Matrix3d(state->_calib_IMUtoCAM[1]->Rot());
  auto p_IinC0 = Eigen::Vector3d(state->_calib_IMUtoCAM[0]->pos());
  auto p_IinC1 = Eigen::Vector3d(state->_calib_IMUtoCAM[1]->pos());

  // criteria point : {global}
  Eigen::Matrix3d R_GtoC0_prev = R_ItoC0 * R_GtoI_prev;
  Eigen::Matrix3d R_GtoC0_now = R_ItoC0 * R_GtoI_now;
  Eigen::Matrix3d R_GtoC1_prev = R_ItoC1 * R_GtoI_prev;
  Eigen::Matrix3d R_GtoC1_now = R_ItoC1 * R_GtoI_now;
  Eigen::Vector3d p_GinC0_prev = p_IinC0 - R_GtoC0_prev * p_IinG_prev;
  Eigen::Vector3d p_GinC0_now = p_IinC0 - R_GtoC0_now * p_IinG_now;
  Eigen::Vector3d p_GinC1_prev = p_IinC1 - R_GtoC1_prev * p_IinG_prev;
  Eigen::Vector3d p_GinC1_now = p_IinC1 - R_GtoC1_now * p_IinG_now;

  // criteria point : {cam0}
  Eigen::Matrix3d R_C0toCO_prev = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_C0toC0_now = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_C0toC1_prev = R_GtoC1_prev * R_GtoC0_prev.transpose();
  R_C0toC1_now = R_GtoC1_now * R_GtoC0_now.transpose();
  Eigen::Vector3d p_C0inC0_prev = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_C0inC0_now = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_C0inC1_prev = p_GinC1_prev - R_C0toC1_prev * p_GinC0_prev;
  p_C0inC1_now = p_GinC1_now - R_C0toC1_now * p_GinC0_now;
// printf("extrinsic before---------------------\n");
// std::cout<<R_C0BtoC1_prev<<std::endl;
// std::cout<<p_C0BinC1_prev<<std::endl;
// printf("extrinsic after---------------------\n");
// std::cout<<R_C0AtoC1_now<<std::endl;
// std::cout<<p_C0AinC1_now<<std::endl;

  //Since I've been passed normalized points, the projection matrix is an extrinsic
  Eigen::Matrix<double, 3, 4> P_C0_prev_temp;
  Eigen::Matrix<double, 3, 4> P_C0_now_temp;
  Eigen::Matrix<double, 3, 4> P_C1_prev_temp;
  Eigen::Matrix<double, 3, 4> P_C1_now_temp;
  P_C0_prev_temp << R_C0toCO_prev, p_C0inC0_prev;
  P_C0_now_temp << R_C0toC0_now, p_C0inC0_now;
  P_C1_prev_temp << R_C0toC1_prev, p_C0inC1_prev;
  P_C1_now_temp << R_C0toC1_now, p_C0inC1_now;
  cv::Mat P_C0_prev;
  cv::Mat P_C0_now;
  cv::Mat P_C1_prev;
  cv::Mat P_C1_now;
  cv::eigen2cv(P_C0_prev_temp, P_C0_prev);
  cv::eigen2cv(P_C0_now_temp, P_C0_now);
  cv::eigen2cv(P_C1_prev_temp, P_C1_prev);
  cv::eigen2cv(P_C1_now_temp, P_C1_now);
  auto &all_p2ds_C0_prev = all_p2ds_set_C0[0];
  auto &all_p2ds_C1_prev = all_p2ds_set_C1[0];
  auto &all_p2ds_C0_now = all_p2ds_set_C0[1];
  auto &all_p2ds_C1_now = all_p2ds_set_C1[1];
  cv::triangulatePoints(P_C0_prev, P_C1_prev, all_p2ds_C0_prev, all_p2ds_C1_prev, all_p3ds_prev);
  cv::triangulatePoints(P_C0_now, P_C1_now, all_p2ds_C0_now, all_p2ds_C1_now, all_p3ds_now);

  // Compute a prediction of the points for motion, assuming the received points are static
  Eigen::Matrix3d R_C0_prev_to_now = R_GtoC0_now * R_GtoC0_prev.transpose();
  Eigen::Vector3d p_C0_prev_to_now = p_GinC0_now - R_C0_prev_to_now * p_GinC0_prev;
  Eigen::Matrix<double, 3, 4> P_C0_prev_to_now;
  P_C0_prev_to_now << R_C0_prev_to_now, p_C0_prev_to_now;
  Eigen::Matrix<double, 1, 4> T_bottom;
  T_bottom << 0.0, 0.0, 0.0, 1.0;
  T_C0_prev_to_now << P_C0_prev_to_now, T_bottom;
  cv::Mat T_C0_prev_to_now_cv;
  cv::eigen2cv(T_C0_prev_to_now, T_C0_prev_to_now_cv);
  all_p3ds_prev.convertTo(all_p3ds_prev, CV_64F);
  all_p3ds_now.convertTo(all_p3ds_now, CV_64F);
  all_p3ds_pred = T_C0_prev_to_now_cv * all_p3ds_prev;
}

bool ObjectTracker::get_sampled_idcs(std::vector<size_t> &sampled_idcs, std::vector<size_t> &rnd_base, const std::vector<size_t> &idcs, const size_t k, std::set<std::vector<size_t>> &used) {
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

int ObjectTracker::get_max_iters(const double prob, const double outliers_ratio, const size_t k, const size_t num_idcs) {
  auto it = iter_table.find(num_idcs);
  if (it == iter_table.end()) {
    int max_iters = std::log(1 - prob) / std::log(1 - std::pow(1 - outliers_ratio, k));
    if (num_idcs < 10) {
      max_iters = std::min(max_iters, nCr(num_idcs, k));
    }

    iter_table.emplace(num_idcs, max_iters);
    return max_iters;
  }
  
  return it->second;
}

bool ObjectTracker::pass_svd(std::vector<uchar> &mask_out, const std::vector<size_t> &idcs, const std::vector<float> &L2_3d_vec, const int best_num_consensus, Eigen::Matrix4f &inliers_tf) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_prev_temp(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_now_temp(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<float> L2_3d_vec_temp;
  for (size_t full_idx = 0; full_idx<mask_out.size(); ++full_idx) {
    if (mask_out[full_idx] == 0)
      continue;

    size_t p3ds_idx = idcs[full_idx];
    inliers_prev_temp->emplace_back(p3ds_prev->at(p3ds_idx));
    inliers_now_temp->emplace_back(p3ds_now->at(p3ds_idx));
    L2_3d_vec_temp.emplace_back(L2_3d_vec[full_idx]);
  }

  std::vector<float> sorting_idx(L2_3d_vec_temp);
  std::iota(sorting_idx.begin(), sorting_idx.end(), 0);
  std::sort(sorting_idx.begin(), sorting_idx.end(), 
            [&](float idx0, float idx1) {return L2_3d_vec_temp[idx0] < L2_3d_vec_temp[idx1];});

  int max_elements = std::min(best_num_consensus, MAX_NUM_SVD);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_prev(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_now(new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<float> partial_L2_3d_vec;
  inliers_prev->resize(max_elements);
  inliers_now->resize(max_elements);
  partial_L2_3d_vec.resize(max_elements);
  for (size_t idx_to = 0; idx_to<(size_t)max_elements; ++idx_to)
  {
    size_t idx_from = sorting_idx[idx_to];
    inliers_prev->at(idx_to) = inliers_prev_temp->at(idx_from);
    inliers_now->at(idx_to) = inliers_now_temp->at(idx_from);
    partial_L2_3d_vec[idx_to] = L2_3d_vec_temp[idx_from];
  }

  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
  svd.estimateRigidTransformation(*inliers_prev, *inliers_now, inliers_tf);
  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pred_p3ds(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*inliers_prev, *inliers_pred_p3ds, inliers_tf);

  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pred_ego(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f T_C0BtoC0A_f = T_C0_prev_to_now.cast<float>();
  pcl::transformPointCloud(*inliers_prev, *inliers_pred_ego, T_C0BtoC0A_f);

  double weighted_L2_3d = 0.0;
  double L2_2d = 0.0;
  double z_mean = 0.0;
  for (size_t i = 0; i<(size_t)max_elements; ++i) {
    Eigen::Vector3f weighted_diff_3d = inliers_pred_p3ds->at(i).getVector3fMap() - inliers_pred_ego->at(i).getVector3fMap();
    weighted_diff_3d.block<1, 1>(2, 0) *= Z_PENALTY;
    weighted_L2_3d += (double)weighted_diff_3d.norm();
    double u_pred_ego = (double)inliers_pred_ego->at(i).x/(double)inliers_pred_ego->at(i).z;
    double v_pred_ego = (double)inliers_pred_ego->at(i).y/(double)inliers_pred_ego->at(i).z;
    double u_pred_p2ds = (double)inliers_pred_p3ds->at(i).x/(double)inliers_pred_p3ds->at(i).z;
    double v_pred_p2ds = (double)inliers_pred_p3ds->at(i).y/(double)inliers_pred_p3ds->at(i).z;
    auto pt_pred_ego = cv::Point2d(u_pred_ego, v_pred_ego);
    auto pt_pred_p2ds = cv::Point2d(u_pred_p2ds, v_pred_p2ds);
    cv::Point2d diff_2d = pt_pred_ego-pt_pred_p2ds;
    L2_2d += cv::norm(diff_2d);
    z_mean += inliers_pred_p3ds->at(i).z;
  }
  weighted_L2_3d /= max_elements;
  L2_2d *= fx0/max_elements;
  z_mean /= max_elements;
  double th_3d = ALPHA + BETA/max_elements;
  double th_2d = DELTA/z_mean + GAMMA/max_elements;

printf("3d : %f / %f\n", weighted_L2_3d, th_3d);
printf("2d : %f / %f\n", L2_2d, th_2d);
printf("is_dynamic? %d\n", z_mean > 0.0 && weighted_L2_3d > th_3d && L2_2d > th_2d);
  if(z_mean > 0.0 && weighted_L2_3d > th_3d && L2_2d > th_2d) {
    return true;
  }

  return false;
}

bool ObjectTracker::pass_ransac(const std::vector<size_t> &idcs, const float th_L2_3d, const double prob_ransac, std::vector<uchar> &mask_out, std::vector<float> &L2_3d_vec, int &best_num_consensus) {
printf("input : ");
for(int i = 0;i<idcs.size(); ++i)
printf("%d ", idcs[i]);
printf("\n");
  // Initialize the RANSAC parameters
  // Number of points to draw for each iteration
  const size_t k = MIN_NUM_TF; 
  size_t num_idcs = idcs.size();
  int max_iters = get_max_iters(prob_ransac, OUTLIERS_RATIO, k, num_idcs);
printf("max_iters %d \n", max_iters);
  best_num_consensus = -1;
  std::vector<uchar> consensus_mask;
  std::vector<size_t> sampled_idcs;
  std::vector<size_t> rnd_base(idcs.size());
  std::iota(rnd_base.begin(), rnd_base.end(), 0);
  std::set<std::vector<size_t>> used;

  // Iterate for a maximum number of iterations
  int fail_cnt = 0;
  for (int iter = 0; iter < max_iters && fail_cnt<50; iter++) {
    bool sample_succeeded = get_sampled_idcs(sampled_idcs, rnd_base, idcs, k, used);
    if (!sample_succeeded) {
        --iter;
        ++fail_cnt;
        continue;
    }

    // Estimate the transformation using the k random points
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_p3ds_prev(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_p3ds_now(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i<k; ++i) {
      sampled_p3ds_prev->emplace_back(p3ds_prev->at(sampled_idcs[i]));
      sampled_p3ds_now->emplace_back(p3ds_now->at(sampled_idcs[i]));
    }
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    Eigen::Matrix4f sampled_p3ds_tf;
    svd.estimateRigidTransformation(*sampled_p3ds_prev, *sampled_p3ds_now, sampled_p3ds_tf);

    // Evaluate the transformation using the available points
    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_p2ds_pred(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*p3ds_prev, *sampled_p2ds_pred, sampled_p3ds_tf);

    int num_consensus = 0;
    consensus_mask = std::vector<uchar>(idcs.size(), 0);
    L2_3d_vec = std::vector<float>(idcs.size(), std::numeric_limits<float>::max());
    for (size_t i = 0; i<num_idcs; ++i) {
      float sampled_L2_3d = (sampled_p2ds_pred->at(idcs[i]).getVector3fMap() - p3ds_now->at(idcs[i]).getVector3fMap()).norm();
      if (sampled_L2_3d < th_L2_3d) {
        ++num_consensus;
        consensus_mask[i] = IS_INLIER;
        L2_3d_vec[i] = sampled_L2_3d;
      }
    }

    // Update the best transformation if the current transformation has more inliers
    if (num_consensus > best_num_consensus) {
      best_num_consensus = num_consensus;
      mask_out = consensus_mask;
    }
  }
  
printf("res (%d ea) :", best_num_consensus);
for(int i = 0; i<mask_out.size(); ++i)
{
  if(mask_out[i] == 1)
    printf("%d ", idcs[i]);
}
std::cout<<std::endl;

  if (best_num_consensus >= k)
    return true;

  return false;
}

void ObjectTracker::optimize(const Eigen::Matrix4f &inliers_tf, const std::vector<uchar> &mask, const std::vector<size_t> &idcs) {
  Eigen::Quaterniond q_inliers(inliers_tf.block<3,3>(0,0).cast<double>());

  // Initialize problem
  ceres::Problem problem;

  // Add parameter block for quaternion rotation
  double q_para[4];
  q_para[0] = q_inliers.w();
  q_para[1] = q_inliers.x();
  q_para[2] = q_inliers.y();
  q_para[3] = q_inliers.z();
  problem.AddParameterBlock(q_para, 4, new ceres::EigenQuaternionManifold);
  
  // Add parameter block for translation
  double p_para[3];
  p_para[0] = inliers_tf(0,3);
  p_para[1] = inliers_tf(1,3);
  p_para[2] = inliers_tf(2,3);
  problem.AddParameterBlock(p_para, 3);

  std::vector<size_t> compact_idcs;
  for (size_t i = 0; i<mask.size(); ++i) {
    if (mask[i] == IS_OUTLIER) 
      continue;

    compact_idcs.emplace_back(idcs[i]);
  }

  std::vector<Eigen::Vector3d> p3ds_para;
  for (size_t i = 0; i<compact_idcs.size(); ++i) {
    size_t idx = compact_idcs[i];
    Eigen::Vector3d p3d_para;
    p3d_para << (double)p3ds_prev->at(idx).x, (double)p3ds_prev->at(idx).y, (double)p3ds_prev->at(idx).z;
    p3ds_para.emplace_back(p3d_para);
  }
  for (size_t i = 0; i<compact_idcs.size(); ++i) {
    problem.AddParameterBlock(p3ds_para[i].data(), 3);
  }

  // Add cost function for each observation
  for (size_t i = 0; i<compact_idcs.size(); ++i) {
    size_t idx = compact_idcs[i];
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<ReprojectionErrorCostFunction, 8, 3, 4, 3>(
            new ReprojectionErrorCostFunction(p2ds_C0_prev[idx], p2ds_C1_prev[idx],
                                              p2ds_C0_now[idx], p2ds_C1_now[idx],
                                              Eigen::Quaterniond(R_C0toC1_now), p_C0inC1_now,
                                              fx0, fy0));
    problem.AddResidualBlock(cost_function, nullptr, p3ds_para[i].data(), q_para, p_para);
  }

  // Set solver options
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Quaterniond q_refine(q_para[0], q_para[1], q_para[2], q_para[3]);
  Eigen::Vector3d p_refine(p_para[0], p_para[1], p_para[2]);
  for(size_t i = 0; i<p3ds_para.size(); ++i)
  {
    Eigen::Vector3d p3d_refine = q_refine * p3ds_para[i] + p_refine;
    size_t idx = compact_idcs[i];
    p3ds_now->at(idx) = pcl::PointXYZ((float)p3d_refine.x(), (float)p3d_refine.y(), (float)p3d_refine.z());
    p2ds_C0_now[idx] = Eigen::Vector2d(p3d_refine.x()/p3d_refine.z(), p3d_refine.y()/p3d_refine.z());
  }
}

void ObjectTracker::refine_dynamic(const std::vector<size_t> &idcs, bool &succeeded, bool &dynamic, std::vector<uchar> &mask, Eigen::Matrix4f &inliers_tf) {
  std::vector<float> L2_3d_vec;
  int best_num_consensus;
  succeeded = pass_ransac(idcs, TH_L2_3D_RANSAC, PROB_RANSAC, mask, L2_3d_vec, best_num_consensus);
  if (!succeeded)
    return;

  dynamic = false;
  dynamic = pass_svd(mask, idcs, L2_3d_vec, best_num_consensus, inliers_tf);
  if (!dynamic)
    return;

  optimize(inliers_tf, mask, idcs);
}

void ObjectTracker::label_p3ds(std::vector<std::vector<std::vector<size_t>>> &labeled_idcs_2d, std::vector<std::vector<uchar>> &labeled_states_2d, const std::vector<std::vector<size_t>> &graphed_idcs) {
  labeled_idcs_2d = std::vector<std::vector<std::vector<size_t>>>(graphed_idcs.size());
  labeled_states_2d = std::vector<std::vector<uchar>>(graphed_idcs.size());
  size_t num_graph = graphed_idcs.size();
  for (size_t graph_idx = 0; graph_idx<num_graph; ++graph_idx) {
    bool succeeded = true;
    bool dinamic = false;
    std::vector<size_t> idcs_now = graphed_idcs[graph_idx];
    std::vector<uchar> mask(idcs_now.size(), IS_OUTLIER);
    Eigen::Matrix4f tf_now;
    refine_dynamic(idcs_now, succeeded, dinamic, mask, tf_now);
    while (succeeded) {
      std::vector<size_t> idcs_next;
      std::vector<size_t> inlier_idcs;
      for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == IS_INLIER) {
          inlier_idcs.emplace_back(idcs_now[i]);
        }
        else {
          idcs_next.emplace_back(idcs_now[i]);
        }
      }

      labeled_idcs_2d[graph_idx].emplace_back(inlier_idcs);
      if (dinamic) {
        labeled_states_2d[graph_idx].emplace_back(DYNAMIC);
      }
      else {
        labeled_states_2d[graph_idx].emplace_back(STATIC);
      }

printf("next suggestion : ");
for(int i = 0; i<idcs_next.size(); ++i)
printf("%d ", idcs_next[i]);
printf("\n");

      if (idcs_next.size() < MIN_NUM_TF) {
        labeled_idcs_2d[graph_idx].emplace_back(idcs_next);
        labeled_states_2d[graph_idx].emplace_back(IS_OUTLIER);
        break;
      }

      idcs_now = idcs_next;
      mask = std::vector<uchar>(idcs_next.size(), IS_OUTLIER);
      Eigen::Matrix4f tf_next;
      refine_dynamic(idcs_next, succeeded, dinamic, mask, tf_next);

      if (!succeeded) {
        labeled_idcs_2d[graph_idx].emplace_back(idcs_next);
        labeled_states_2d[graph_idx].emplace_back(IS_OUTLIER);
      } 
    }
  }
}

void ObjectTracker::init(const std::shared_ptr<ov_msckf::State> &state) {
  p2ds_C0_now.clear();
  p2ds_C0_prev.clear();
  p2ds_C1_now.clear();
  p2ds_C1_prev.clear();
  p3ds_now = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  p3ds_now->clear();
  p3ds_prev = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  p3ds_prev->clear();
  raw_idcs.clear();

  fx0 = state->_cam_intrinsics.at(0)->value()(0);
  cx0 = state->_cam_intrinsics.at(0)->value()(2);
  fy0 = state->_cam_intrinsics.at(0)->value()(1);
  cy0 = state->_cam_intrinsics.at(0)->value()(3);
  fx1 = state->_cam_intrinsics.at(1)->value()(0);
  cx1 = state->_cam_intrinsics.at(1)->value()(2);
  fy1 = state->_cam_intrinsics.at(1)->value()(1);
  cy1 = state->_cam_intrinsics.at(1)->value()(3);
}

void ObjectTracker::rm_outliers(const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C0, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C1, const std::vector<size_t> &all_raw_idcs) {
  cv::Mat all_p3ds_prev;
  cv::Mat all_p3ds_now;
  cv::Mat all_p3ds_pred;
  get_meas_and_pred(state, all_p2ds_set_C0, all_p2ds_set_C1, all_p3ds_prev, all_p3ds_now, all_p3ds_pred);

  // reject static pts, get only is_dynamic pts
  reject_static_p3ds(all_p3ds_prev, all_p3ds_now, all_p3ds_pred, all_p2ds_set_C0, all_p2ds_set_C1, all_raw_idcs);
}

void ObjectTracker::divide_graphs(std::vector<std::vector<size_t>> &graphed_idcs, std::vector<std::vector<size_t>> &labeled_idcs, std::vector<std::vector<size_t>> &labeled_raw_idcs) {
  std::vector<std::vector<std::vector<size_t>>> labeled_idcs_2d;
  std::vector<std::vector<uchar>> labeled_states_2d;
  label_p3ds(labeled_idcs_2d, labeled_states_2d, graphed_idcs);

  for (size_t graph_idx = 0; graph_idx<labeled_idcs_2d.size(); ++graph_idx) {
    for (size_t label_idx = 0; label_idx<labeled_idcs_2d[graph_idx].size(); ++label_idx) {
      if(labeled_states_2d[graph_idx][label_idx] != DYNAMIC)
        continue;

      std::vector<size_t> partial_raw_idcs;
      for (size_t pt_idx = 0; pt_idx<labeled_idcs_2d[graph_idx][label_idx].size(); ++pt_idx) {
        size_t idx = labeled_idcs_2d[graph_idx][label_idx][pt_idx];
        partial_raw_idcs.emplace_back(raw_idcs[idx]);
      }
      labeled_raw_idcs.emplace_back(partial_raw_idcs);
      labeled_idcs.emplace_back(labeled_idcs_2d[graph_idx][label_idx]);
    }
  }

printf("labels\n");
for(int i = 0; i<labeled_idcs_2d.size(); ++i)
{
printf("graphs %d :\n", i);
for(int j = 0; j<labeled_idcs_2d[i].size(); ++j)
{
auto label_color = get_random_color();
if(labeled_states_2d[i][j] != 1)
label_color = cv::Scalar(255,255,255);
printf("{");
for(size_t k = 0; k<labeled_idcs_2d[i][j].size(); ++k)
{
printf("%d ", labeled_idcs_2d[i][j][k]);
int idx = labeled_idcs_2d[i][j][k];
auto q = cv::Point2d(fx0*p2ds_C0_now[idx].x()+cx0, fy0*p2ds_C0_now[idx].y()+cy0);
cv::circle(test_l, q, 3, label_color, 3);
// if(labeled_states_2d[i][j] == 1)
// {
// cv::circle(test_l, q, 13, cv::Scalar(0,255,0), 2);
// std::string s = std::to_string(raw_idcs[idx]);
// cv::putText(test_l, s, q, 1, 1, cv::Scalar(0, 255, 0));
// }
}
printf("}(%d) ", labeled_states_2d[i][j]);
}
printf("\n");
}

}

void ObjectTracker::get_pair(const size_t num_labels_now, const std::vector<std::vector<size_t>> &labeled_raw_idcs, std::vector<std::pair<size_t, size_t>> &pairs) {
  size_t mat_size = std::max(num_labels_prev, num_labels_now);
  std::vector<std::vector<int>> cost_matrix(mat_size, std::vector<int>(mat_size, 0));
  int min_val = 0;
  for (size_t label_idx_now = 0; label_idx_now<num_labels_now; ++label_idx_now) {
    for (size_t pt_idx = 0; pt_idx < labeled_raw_idcs[label_idx_now].size(); ++pt_idx) {
      size_t raw_idx_now = labeled_raw_idcs[label_idx_now][pt_idx];
      auto it_prev = raw_idcs_table_prev.find(raw_idx_now);
      if(it_prev == raw_idcs_table_prev.end())
        continue;

      size_t label_idx_prev = it_prev->second;
      --cost_matrix[label_idx_now][label_idx_prev];
      min_val = std::min(min_val, cost_matrix[label_idx_now][label_idx_prev]);
    }
  }

  for(size_t i = 0; i<cost_matrix.size(); ++i) {
    for(int j = 0; j<cost_matrix[i].size(); ++j) {
      cost_matrix[i][j] -= min_val;
    }
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

  get_hungarian_pairs(cost_matrix, pairs);
  std::vector<std::pair<size_t, size_t>> pairs_temp;
printf("pairs\n");
  for(int i = 0; i<pairs.size(); ++i) {
    if(cost_matrix[pairs[i].first][pairs[i].second] >= -min_val)
      continue;
  
    pairs_temp.emplace_back(pairs[i]);
printf("%d -> %d\n", pairs[i].first, pairs[i].second);
  }
  pairs = pairs_temp;

}

void ObjectTracker::track(const ov_core::CameraData &message, const std::shared_ptr<ov_msckf::State> &state, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C0, const std::vector<std::vector<cv::Point2f>> &all_p2ds_set_C1, const std::vector<size_t> &all_raw_idcs, std::unordered_set<size_t> &tracked_idcs) {

cv::cvtColor(message.images.at(0), test_l, cv::COLOR_GRAY2RGB);

  tracked_idcs.clear();

  size_t num_labels_now = 0;
  std::unordered_map<size_t, size_t> raw_idcs_table_now;
  std::map<size_t, cv::Scalar> tracked_labels_table_now;

  if (all_p2ds_set_C0[0].empty()) {
cv::imshow("test", test_l);
cv::waitKey(1);

    raw_idcs_table_prev = raw_idcs_table_now;
    num_labels_prev = num_labels_now;
    tracked_labels_table_prev = tracked_labels_table_now;
    R_GtoI_prev = Eigen::Matrix3d(state->_imu->Rot());
    p_IinG_prev = Eigen::Vector3d(state->_imu->pos());
    return;
  }

  init(state);
  ObjectTracker::rm_outliers(state, all_p2ds_set_C0, all_p2ds_set_C1, all_raw_idcs);

  // Get k nearest points in the neighbourhood
  if (p3ds_now->size() > NUM_NEAREST) {
    std::vector<std::vector<size_t>> graphs;
    make_graphs(graphs);

    // DFS to bring all elements of a graphs together and reject graphs with fewer nodes
    std::vector<std::vector<size_t>> graphed_idcs;
    serialize_graphs(graphs, graphed_idcs);

for(int i = 0; i<graphed_idcs.size(); ++i) {
for(int j = 0; j <graphed_idcs[i].size(); ++j) {
printf("%d ", graphed_idcs[i][j]);
}
std::cout<<std::endl;
}

    std::vector<std::vector<size_t>> labeled_idcs;
    std::vector<std::vector<size_t>> labeled_raw_idcs;
    divide_graphs(graphed_idcs, labeled_idcs, labeled_raw_idcs);
    
    num_labels_now = labeled_raw_idcs.size();
    for (size_t label_idx = 0; label_idx<num_labels_now; ++label_idx) {
      for (size_t pt_idx = 0; pt_idx<labeled_raw_idcs[label_idx].size(); ++pt_idx) {
        size_t raw_idx = labeled_raw_idcs[label_idx][pt_idx];
        raw_idcs_table_now.emplace(raw_idx, label_idx);
        tracked_idcs.emplace(raw_idx);
      }
      tracked_labels_table_now.emplace(label_idx, get_random_color());
    }

    std::vector<std::pair<size_t, size_t>> pairs;
    if (num_labels_now > 0 && num_labels_prev > 0) {
      get_pair(num_labels_now, labeled_raw_idcs, pairs);
    }

for (size_t i = 0; i<pairs.size(); ++i) {
  size_t label_idx_now = pairs[i].first;
  size_t label_idx_prev = pairs[i].second;

  double x0 = std::numeric_limits<double>::max();
  double x1 = -std::numeric_limits<double>::max();
  double y0 = std::numeric_limits<double>::max();
  double y1 = -std::numeric_limits<double>::max();
  for (size_t pt_idx = 0; pt_idx<labeled_idcs[label_idx_now].size(); ++pt_idx) {
    size_t idx = labeled_idcs[label_idx_now][pt_idx];
    x0 = std::min(x0, p2ds_C0_now[idx].x());
    x1 = std::max(x1, p2ds_C0_now[idx].x());
    y0 = std::min(y0, p2ds_C0_now[idx].y());
    y1 = std::max(y1, p2ds_C0_now[idx].y());
  }
  auto q0 = cv::Point2d(fx0*x0+cx0, fy0*y0+cy0);
  auto q1 = cv::Point2d(fx0*x1+cx0, fy0*y1+cy0);

  auto color = tracked_labels_table_prev[label_idx_prev];
  tracked_labels_table_now[label_idx_now] = color;
  cv::rectangle(test_l, cv::Rect2d(q0, q1), color, 3);
} 


  }

cv::imshow("test", test_l);
cv::waitKey(1);

  raw_idcs_table_prev = raw_idcs_table_now;
  num_labels_prev = num_labels_now;
  tracked_labels_table_prev = tracked_labels_table_now;
  R_GtoI_prev = Eigen::Matrix3d(state->_imu->Rot());
  p_IinG_prev = Eigen::Vector3d(state->_imu->pos());
}