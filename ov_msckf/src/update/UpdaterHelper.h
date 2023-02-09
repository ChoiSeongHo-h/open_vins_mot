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

#ifndef OV_MSCKF_UPDATER_HELPER_H
#define OV_MSCKF_UPDATER_HELPER_H

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>

#include "types/LandmarkRepresentation.h"

namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief Class that has helper functions for our updaters.
 *
 * Can compute the Jacobian for a single feature representation.
 * This will create the Jacobian based on what representation our state is in.
 * If we are using the anchor representation then we also have additional Jacobians in respect to the anchor state.
 * Also has functions such as nullspace projection and full jacobian construction.
 * For derivations look at @ref update-feat page which has detailed equations.
 *
 */
class UpdaterHelper {
public:
  /**
   * @brief Feature object that our UpdaterHelper leverages, has all measurements and means
   */
  struct UpdaterHelperFeature {

    /// Unique ID of this feature
    size_t featid;

    /// UV coordinates that this feature has been seen from (mapped by camera ID)
    std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs;

    // UV normalized coordinates that this feature has been seen from (mapped by camera ID)
    std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs_norm;

    /// Timestamps of each UV measurement (mapped by camera ID)
    std::unordered_map<size_t, std::vector<double>> timestamps;

    /// What representation our feature is in
    ov_type::LandmarkRepresentation::Representation feat_representation;

    /// What camera ID our pose is anchored in!! By default the first measurement is the anchor.
    int anchor_cam_id = -1;

    /// Timestamp of anchor clone
    double anchor_clone_timestamp = -1;

    /// Triangulated position of this feature, in the anchor frame
    Eigen::Vector3d p_FinA;

    /// Triangulated position of this feature, in the anchor frame first estimate
    Eigen::Vector3d p_FinA_fej;

    /// Triangulated position of this feature, in the global frame
    Eigen::Vector3d p_FinG;

    /// Triangulated position of this feature, in the global frame first estimate
    Eigen::Vector3d p_FinG_fej;
  };

  /**
   * @brief This gets the feature and state Jacobian in respect to the feature representation
   *
   * @param[in] state State of the filter system
   * @param[in] feature Feature we want to get Jacobians of (must have feature means)
   * @param[out] H_f Jacobians in respect to the feature error state (will be either 3x3 or 3x1 for single depth)
   * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
   * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
   */
  static void get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                  std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief Will construct the "stacked" Jacobians for a single feature from all its measurements
   *
   * @param[in] state State of the filter system
   * @param[in] feature Feature we want to get Jacobians of (must have feature means)
   * @param[out] H_f Jacobians in respect to the feature error state
   * @param[out] H_x Extra Jacobians in respect to the state (for example anchored pose)
   * @param[out] res Measurement residual for this feature
   * @param[out] x_order Extra variables our extra Jacobian has (for example anchored pose)
   */
  static void get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                        Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief This will project the left nullspace of H_f onto the linear system.
   *
   * Please see the @ref update-null for details on how this works.
   * This is the MSCKF nullspace projection which removes the dependency on the feature state.
   * Note that this is done **in place** so all matrices will be different after a function call.
   *
   * @param H_f Jacobian with nullspace we want to project onto the system [res = Hx*(x-xhat)+Hf(f-fhat)+n]
   * @param H_x State jacobian
   * @param res Measurement residual
   */
  static void nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res);

  /**
   * @brief This will perform measurement compression
   *
   * Please see the @ref update-compress for details on how this works.
   * Note that this is done **in place** so all matrices will be different after a function call.
   *
   * @param H_x State jacobian
   * @param res Measurement residual
   */
  static void measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res);
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_HELPER_H