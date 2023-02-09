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

#include "ResultSimulation.h"

using namespace ov_eval;

ResultSimulation::ResultSimulation(std::string path_est, std::string path_std, std::string path_gt) {

  // Load from disk
  Loader::load_simulation(path_est, est_state);
  Loader::load_simulation(path_std, state_cov);
  Loader::load_simulation(path_gt, gt_state);

  /// Assert they are of equal length
  assert(est_state.size() == state_cov.size());
  assert(est_state.size() == gt_state.size());

  // Debug print
  PRINT_DEBUG("[SIM]: loaded %d timestamps from file!!\n", (int)est_state.size());
  PRINT_DEBUG("[SIM]: we have %d cameras in total!!\n", (int)est_state.at(0)(18));
}

void ResultSimulation::plot_state(bool doplotting, double max_time) {

  // Errors for each xyz direction
  Statistics error_ori[3], error_pos[3], error_vel[3], error_bg[3], error_ba[3];

  // Calculate the position and orientation error at every timestep
  double start_time = est_state.at(0)(0);
  for (size_t i = 0; i < est_state.size(); i++) {

    // Exit if we have reached our max time
    if ((est_state.at(i)(0) - start_time) > max_time)
      break;

    // Assert our times are the same
    assert(est_state.at(i)(0) == gt_state.at(i)(0));

    // Calculate orientation error
    // NOTE: we define our error as e_R = -Log(R*Rhat^T)
    Eigen::Matrix3d e_R =
        ov_core::quat_2_Rot(gt_state.at(i).block(1, 0, 4, 1)) * ov_core::quat_2_Rot(est_state.at(i).block(1, 0, 4, 1)).transpose();
    Eigen::Vector3d ori_err = -180.0 / M_PI * ov_core::log_so3(e_R);
    for (int j = 0; j < 3; j++) {
      error_ori[j].timestamps.push_back(est_state.at(i)(0));
      error_ori[j].values.push_back(ori_err(j));
      error_ori[j].values_bound.push_back(3 * 180.0 / M_PI * state_cov.at(i)(1 + j));
      error_ori[j].calculate();
    }

    // Calculate position error
    Eigen::Vector3d pos_err = gt_state.at(i).block(5, 0, 3, 1) - est_state.at(i).block(5, 0, 3, 1);
    for (int j = 0; j < 3; j++) {
      error_pos[j].timestamps.push_back(est_state.at(i)(0));
      error_pos[j].values.push_back(pos_err(j));
      error_pos[j].values_bound.push_back(3 * state_cov.at(i)(4 + j));
      error_pos[j].calculate();
    }

    // Calculate velocity error
    Eigen::Vector3d vel_err = gt_state.at(i).block(8, 0, 3, 1) - est_state.at(i).block(8, 0, 3, 1);
    for (int j = 0; j < 3; j++) {
      error_vel[j].timestamps.push_back(est_state.at(i)(0));
      error_vel[j].values.push_back(vel_err(j));
      error_vel[j].values_bound.push_back(3 * state_cov.at(i)(7 + j));
      error_vel[j].calculate();
    }

    // Calculate gyro bias error
    Eigen::Vector3d bg_err = gt_state.at(i).block(11, 0, 3, 1) - est_state.at(i).block(11, 0, 3, 1);
    for (int j = 0; j < 3; j++) {
      error_bg[j].timestamps.push_back(est_state.at(i)(0));
      error_bg[j].values.push_back(bg_err(j));
      error_bg[j].values_bound.push_back(3 * state_cov.at(i)(10 + j));
      error_bg[j].calculate();
    }

    // Calculate accel bias error
    Eigen::Vector3d ba_err = gt_state.at(i).block(14, 0, 3, 1) - est_state.at(i).block(14, 0, 3, 1);
    for (int j = 0; j < 3; j++) {
      error_ba[j].timestamps.push_back(est_state.at(i)(0));
      error_ba[j].values.push_back(ba_err(j));
      error_ba[j].values_bound.push_back(3 * state_cov.at(i)(13 + j));
      error_ba[j].calculate();
    }
  }

  // return if we don't want to plot
  if (!doplotting)
    return;

#ifndef HAVE_PYTHONLIBS
  PRINT_ERROR(RED "Unable to plot the state error, just returning..\n" RESET);
  return;
#else

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(1000, 500);
  plot_3errors(error_ori[0], error_ori[1], error_ori[2], "blue", "red");
  // Update the title and axis labels
  matplotlibcpp::subplot(3, 1, 1);
  matplotlibcpp::title("IMU Orientation Error");
  matplotlibcpp::ylabel("x-error (deg)");
  matplotlibcpp::subplot(3, 1, 2);
  matplotlibcpp::ylabel("y-error (deg)");
  matplotlibcpp::subplot(3, 1, 3);
  matplotlibcpp::ylabel("z-error (deg)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(1000, 500);
  plot_3errors(error_pos[0], error_pos[1], error_pos[2], "blue", "red");
  // Update the title and axis labels
  matplotlibcpp::subplot(3, 1, 1);
  matplotlibcpp::title("IMU Position Error");
  matplotlibcpp::ylabel("x-error (m)");
  matplotlibcpp::subplot(3, 1, 2);
  matplotlibcpp::ylabel("y-error (m)");
  matplotlibcpp::subplot(3, 1, 3);
  matplotlibcpp::ylabel("z-error (m)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(1000, 500);
  plot_3errors(error_vel[0], error_vel[1], error_vel[2], "blue", "red");
  // Update the title and axis labels
  matplotlibcpp::subplot(3, 1, 1);
  matplotlibcpp::title("IMU Velocity Error");
  matplotlibcpp::ylabel("x-error (m/s)");
  matplotlibcpp::subplot(3, 1, 2);
  matplotlibcpp::ylabel("y-error (m/s)");
  matplotlibcpp::subplot(3, 1, 3);
  matplotlibcpp::ylabel("z-error (m/s)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(1000, 500);
  plot_3errors(error_bg[0], error_bg[1], error_bg[2], "blue", "red");
  // Update the title and axis labels
  matplotlibcpp::subplot(3, 1, 1);
  matplotlibcpp::title("IMU Gyroscope Bias Error");
  matplotlibcpp::ylabel("x-error (rad/s)");
  matplotlibcpp::subplot(3, 1, 2);
  matplotlibcpp::ylabel("y-error (rad/s)");
  matplotlibcpp::subplot(3, 1, 3);
  matplotlibcpp::ylabel("z-error (rad/s)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(1000, 500);
  plot_3errors(error_ba[0], error_ba[1], error_ba[2], "blue", "red");
  // Update the title and axis labels
  matplotlibcpp::subplot(3, 1, 1);
  matplotlibcpp::title("IMU Accelerometer Bias Error");
  matplotlibcpp::ylabel("x-error (m/s^2)");
  matplotlibcpp::subplot(3, 1, 2);
  matplotlibcpp::ylabel("y-error (m/s^2)");
  matplotlibcpp::subplot(3, 1, 3);
  matplotlibcpp::ylabel("z-error (m/s^2)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

#endif
}

void ResultSimulation::plot_timeoff(bool doplotting, double max_time) {

  // Calculate the time offset error at every timestep
  Statistics error_time;
  double start_time = est_state.at(0)(0);
  for (size_t i = 0; i < est_state.size(); i++) {

    // Exit if we have reached our max time
    if ((est_state.at(i)(0) - start_time) > max_time)
      break;

    // Assert our times are the same
    assert(est_state.at(i)(0) == gt_state.at(i)(0));

    // If we are not calibrating then don't plot it!
    if (state_cov.at(i)(16) == 0.0) {
      PRINT_WARNING(YELLOW "Time offset was not calibrated online, so will not plot...\n" RESET);
      return;
    }

    // Calculate time difference
    error_time.timestamps.push_back(est_state.at(i)(0));
    error_time.values.push_back(est_state.at(i)(17) - gt_state.at(i)(17));
    error_time.values_bound.push_back(3 * state_cov.at(i)(16));
    error_time.calculate();
  }

  // return if we don't want to plot
  if (!doplotting)
    return;

#ifndef HAVE_PYTHONLIBS
  PRINT_ERROR(RED "Matplotlib not loaded, so will not plot, just returning..\n" RESET);
  return;
#else

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(800, 250);

  // Zero our time array
  double starttime = (error_time.timestamps.empty()) ? 0 : error_time.timestamps.at(0);
  double endtime = (error_time.timestamps.empty()) ? 0 : error_time.timestamps.at(error_time.timestamps.size() - 1);
  for (size_t i = 0; i < error_time.timestamps.size(); i++) {
    error_time.timestamps.at(i) -= starttime;
  }

  // Parameters that define the line styles
  std::map<std::string, std::string> params_value, params_bound;
  params_value.insert({"label", "error"});
  params_value.insert({"linestyle", "-"});
  params_value.insert({"color", "blue"});
  params_bound.insert({"label", "3 sigma bound"});
  params_bound.insert({"linestyle", "--"});
  params_bound.insert({"color", "red"});

  // Plot our error value
  matplotlibcpp::plot(error_time.timestamps, error_time.values, params_value);
  if (!error_time.values_bound.empty()) {
    matplotlibcpp::plot(error_time.timestamps, error_time.values_bound, params_bound);
    for (size_t i = 0; i < error_time.timestamps.size(); i++) {
      error_time.values_bound.at(i) *= -1;
    }
    matplotlibcpp::plot(error_time.timestamps, error_time.values_bound, "r--");
  }
  matplotlibcpp::xlim(0.0, endtime - starttime);

  // Update the title and axis labels
  matplotlibcpp::title("Camera IMU Time Offset Error");
  matplotlibcpp::ylabel("error (sec)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

#endif
}

void ResultSimulation::plot_cam_instrinsics(bool doplotting, double max_time) {

  // Check that we have cameras
  if ((int)est_state.at(0)(18) < 1) {
    PRINT_ERROR(YELLOW "You need at least one camera to plot intrinsics...\n" RESET);
    return;
  }

  // Camera intrinsics statistic storage
  std::vector<std::vector<Statistics>> error_cam_k, error_cam_d;
  for (int i = 0; i < (int)est_state.at(0)(18); i++) {
    std::vector<Statistics> temp1, temp2;
    for (int j = 0; j < 4; j++) {
      temp1.push_back(Statistics());
      temp2.push_back(Statistics());
    }
    error_cam_k.push_back(temp1);
    error_cam_d.push_back(temp2);
  }

  // Loop through and calculate error
  double start_time = est_state.at(0)(0);
  for (size_t i = 0; i < est_state.size(); i++) {

    // Exit if we have reached our max time
    if ((est_state.at(i)(0) - start_time) > max_time)
      break;

    // Assert our times are the same
    assert(est_state.at(i)(0) == gt_state.at(i)(0));

    // If we are not calibrating then don't plot it!
    if (state_cov.at(i)(18) == 0.0) {
      PRINT_WARNING(YELLOW "Camera intrinsics not calibrated online, so will not plot...\n" RESET);
      return;
    }

    // Loop through each camera and calculate error
    for (int n = 0; n < (int)est_state.at(0)(18); n++) {
      for (int j = 0; j < 4; j++) {
        error_cam_k.at(n).at(j).timestamps.push_back(est_state.at(i)(0));
        error_cam_k.at(n).at(j).values.push_back(est_state.at(i)(19 + 15 * n + j) - gt_state.at(i)(19 + 15 * n + j));
        error_cam_k.at(n).at(j).values_bound.push_back(3 * state_cov.at(i)(18 + 14 * n + j));
        error_cam_d.at(n).at(j).timestamps.push_back(est_state.at(i)(0));
        error_cam_d.at(n).at(j).values.push_back(est_state.at(i)(19 + 4 + 15 * n + j) - gt_state.at(i)(19 + 4 + 15 * n + j));
        error_cam_d.at(n).at(j).values_bound.push_back(3 * state_cov.at(i)(18 + 4 + 14 * n + j));
      }
    }
  }

  // return if we don't want to plot
  if (!doplotting)
    return;

#ifndef HAVE_PYTHONLIBS
  PRINT_ERROR(RED "Matplotlib not loaded, so will not plot, just returning..\n" RESET);
  return;
#else

  // Plot line colors
  std::vector<std::string> colors = {"blue", "red", "black", "green", "cyan", "magenta"};
  assert(error_cam_k.size() <= colors.size());

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(800, 600);
  for (int n = 0; n < (int)est_state.at(0)(18); n++) {
    std::string estcolor = ((int)est_state.at(0)(18) == 1) ? "blue" : colors.at(n);
    std::string stdcolor = ((int)est_state.at(0)(18) == 1) ? "red" : colors.at(n);
    plot_4errors(error_cam_k.at(n)[0], error_cam_k.at(n)[1], error_cam_k.at(n)[2], error_cam_k.at(n)[3], colors.at(n), stdcolor);
  }

  // Update the title and axis labels
  matplotlibcpp::subplot(4, 1, 1);
  matplotlibcpp::title("Intrinsics Projection Error");
  matplotlibcpp::ylabel("fx (px)");
  matplotlibcpp::subplot(4, 1, 2);
  matplotlibcpp::ylabel("fy (px)");
  matplotlibcpp::subplot(4, 1, 3);
  matplotlibcpp::ylabel("cx (px)");
  matplotlibcpp::subplot(4, 1, 4);
  matplotlibcpp::ylabel("cy (px)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(800, 600);
  for (int n = 0; n < (int)est_state.at(0)(18); n++) {
    std::string estcolor = ((int)est_state.at(0)(18) == 1) ? "blue" : colors.at(n);
    std::string stdcolor = ((int)est_state.at(0)(18) == 1) ? "red" : colors.at(n);
    plot_4errors(error_cam_d.at(n)[0], error_cam_d.at(n)[1], error_cam_d.at(n)[2], error_cam_d.at(n)[3], estcolor, stdcolor);
  }

  // Update the title and axis labels
  matplotlibcpp::subplot(4, 1, 1);
  matplotlibcpp::title("Intrinsics Distortion Error");
  matplotlibcpp::ylabel("d1");
  matplotlibcpp::subplot(4, 1, 2);
  matplotlibcpp::ylabel("d2");
  matplotlibcpp::subplot(4, 1, 3);
  matplotlibcpp::ylabel("d3");
  matplotlibcpp::subplot(4, 1, 4);
  matplotlibcpp::ylabel("d4");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

#endif
}

void ResultSimulation::plot_cam_extrinsics(bool doplotting, double max_time) {

  // Check that we have cameras
  if ((int)est_state.at(0)(18) < 1) {
    PRINT_ERROR(YELLOW "You need at least one camera to plot intrinsics...\n" RESET);
    return;
  }

  // Camera extrinsics statistic storage
  std::vector<std::vector<Statistics>> error_cam_ori, error_cam_pos;
  for (int i = 0; i < (int)est_state.at(0)(18); i++) {
    std::vector<Statistics> temp1, temp2;
    for (int j = 0; j < 3; j++) {
      temp1.push_back(Statistics());
      temp2.push_back(Statistics());
    }
    error_cam_ori.push_back(temp1);
    error_cam_pos.push_back(temp2);
  }

  // Loop through and calculate error
  double start_time = est_state.at(0)(0);
  for (size_t i = 0; i < est_state.size(); i++) {

    // Exit if we have reached our max time
    if ((est_state.at(i)(0) - start_time) > max_time)
      break;

    // Assert our times are the same
    assert(est_state.at(i)(0) == gt_state.at(i)(0));

    // If we are not calibrating then don't plot it!
    if (state_cov.at(i)(26) == 0.0) {
      PRINT_WARNING(YELLOW "Camera extrinsics not calibrated online, so will not plot...\n" RESET);
      return;
    }

    // Loop through each camera and calculate error
    for (int n = 0; n < (int)est_state.at(0)(18); n++) {
      // NOTE: we define our error as e_R = -Log(R*Rhat^T)
      Eigen::Matrix3d e_R = ov_core::quat_2_Rot(gt_state.at(i).block(27 + 15 * n, 0, 4, 1)) *
                            ov_core::quat_2_Rot(est_state.at(i).block(27 + 15 * n, 0, 4, 1)).transpose();
      Eigen::Vector3d ori_err = -180.0 / M_PI * ov_core::log_so3(e_R);
      // Eigen::Matrix3d e_R = Math::quat_2_Rot(est_state.at(i).block(27+15*n,0,4,1)).transpose() *
      // Math::quat_2_Rot(gt_state.at(i).block(27+15*n,0,4,1)); Eigen::Vector3d ori_err = 180.0/M_PI*Math::log_so3(e_R);
      for (int j = 0; j < 3; j++) {
        error_cam_ori.at(n).at(j).timestamps.push_back(est_state.at(i)(0));
        error_cam_ori.at(n).at(j).values.push_back(ori_err(j));
        error_cam_ori.at(n).at(j).values_bound.push_back(3 * 180.0 / M_PI * state_cov.at(i)(26 + 14 * n + j));
        error_cam_pos.at(n).at(j).timestamps.push_back(est_state.at(i)(0));
        error_cam_pos.at(n).at(j).values.push_back(est_state.at(i)(27 + 4 + 15 * n + j) - gt_state.at(i)(27 + 4 + 15 * n + j));
        error_cam_pos.at(n).at(j).values_bound.push_back(3 * state_cov.at(i)(26 + 3 + 14 * n + j));
      }
    }
  }

  // return if we don't want to plot
  if (!doplotting)
    return;

#ifndef HAVE_PYTHONLIBS
  PRINT_ERROR(RED "Matplotlib not loaded, so will not plot, just returning..\n" RESET);
  return;
#else

  // Plot line colors
  std::vector<std::string> colors = {"blue", "red", "black", "green", "cyan", "magenta"};
  assert(error_cam_ori.size() <= colors.size());

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(800, 500);
  for (int n = 0; n < (int)est_state.at(0)(18); n++) {
    std::string estcolor = ((int)est_state.at(0)(18) == 1) ? "blue" : colors.at(n);
    std::string stdcolor = ((int)est_state.at(0)(18) == 1) ? "red" : colors.at(n);
    plot_3errors(error_cam_ori.at(n)[0], error_cam_ori.at(n)[1], error_cam_ori.at(n)[2], colors.at(n), stdcolor);
  }

  // Update the title and axis labels
  matplotlibcpp::subplot(3, 1, 1);
  matplotlibcpp::title("Camera Calibration Orientation Error");
  matplotlibcpp::ylabel("x-error (deg)");
  matplotlibcpp::subplot(3, 1, 2);
  matplotlibcpp::ylabel("y-error (deg)");
  matplotlibcpp::subplot(3, 1, 3);
  matplotlibcpp::ylabel("z-error (deg)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

  //=====================================================
  // Plot this figure
  matplotlibcpp::figure_size(800, 500);
  for (int n = 0; n < (int)est_state.at(0)(18); n++) {
    std::string estcolor = ((int)est_state.at(0)(18) == 1) ? "blue" : colors.at(n);
    std::string stdcolor = ((int)est_state.at(0)(18) == 1) ? "red" : colors.at(n);
    plot_3errors(error_cam_pos.at(n)[0], error_cam_pos.at(n)[1], error_cam_pos.at(n)[2], estcolor, stdcolor);
  }

  // Update the title and axis labels
  matplotlibcpp::subplot(3, 1, 1);
  matplotlibcpp::title("Camera Calibration Position Error");
  matplotlibcpp::ylabel("x-error (m)");
  matplotlibcpp::subplot(3, 1, 2);
  matplotlibcpp::ylabel("y-error (m)");
  matplotlibcpp::subplot(3, 1, 3);
  matplotlibcpp::ylabel("z-error (m)");
  matplotlibcpp::xlabel("dataset time (s)");
  matplotlibcpp::show(false);
  //=====================================================

#endif
}
