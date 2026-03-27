"""
Route B Constructed agent with improved ego localization.

Replaces the original UKF (which drifts to 2-19m error) with a
complementary filter that achieves ~0.39m mean error.

Inherits everything from RouteBConstructedAgent, only overrides:
  - setup(): adds EgoLocalizer initialization
  - tick(): replaces UKF block with localizer.update()

Environment variables:
  LOCALIZER_STRATEGY: 'complementary' (default), 'raw', 'ukf_tuned'
  LOCALIZER_ALPHA: GPS blend weight for complementary filter (default 0.5)
  TARGET_POSE_SOURCE: 'filtered' (default) uses localizer output
  LIDAR_POSE_SOURCE: 'ukf' (default) or 'localizer' — which pose to use for lidar alignment
"""

import os
import math
import numpy as np

from route_b_constructed_b2d_agent import RouteBConstructedAgent
from agents.navigation.local_planner import RoadOption
from team_code.ego_localizer import EgoLocalizer
import team_code.simlingo.transfuser_utils as t_u
from team_code.lidar_utils import lidar_to_ego_coordinate, algin_lidar
from team_code.ukf_utils import bicycle_model_forward
from dataset.generate_lidar_bev_b2d import generate_lidar_bev_images

import cv2
import torch

# TransFuser imports (needed for tick override)
import model.transfuser_extractor.transfuser_utils as transfuser_t_u

IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
LOCALIZER_STRATEGY = os.environ.get('LOCALIZER_STRATEGY', 'complementary').lower()
LOCALIZER_ALPHA = float(os.environ.get('LOCALIZER_ALPHA', '0.5'))
LIDAR_POSE_SOURCE = os.environ.get('LIDAR_POSE_SOURCE', 'ukf').lower()  # 'ukf' or 'localizer'
LOCALIZER_LATENCY_COMPENSATION = os.environ.get('LOCALIZER_LATENCY_COMPENSATION', '0').lower() in (
    '1', 'true', 'yes', 'on'
)
LIDAR_LATENCY_COMPENSATION = os.environ.get('LIDAR_LATENCY_COMPENSATION', '0').lower() in (
    '1', 'true', 'yes', 'on'
)
TARGET_POINT_PROMOTION_DISTANCE_M = 3.0
TARGET_POINT_BEHIND_EGO_EPS_M = 0.5
TARGET_POINT_DEMOTION_DISTANCE_M = 4.0


def get_entry_point():
    return 'RouteBConstructedLocalizerAgent'


class RouteBConstructedLocalizerAgent(RouteBConstructedAgent):

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        # Replace UKF with EgoLocalizer
        self.localizer = EgoLocalizer(
            strategy=LOCALIZER_STRATEGY,
            dt=self.carla_frame_rate,
            gps_alpha=LOCALIZER_ALPHA,
            latency_compensation=LOCALIZER_LATENCY_COMPENSATION,
        )
        print(
            f"[EgoLocalizer] strategy={LOCALIZER_STRATEGY}, alpha={LOCALIZER_ALPHA}, "
            f"target_latency_comp={LOCALIZER_LATENCY_COMPENSATION}, "
            f"lidar_pose={LIDAR_POSE_SOURCE}, lidar_latency_comp={LIDAR_LATENCY_COMPENSATION}"
        )
        self.guard_target_point_world = None
        self.guard_next_target_point_world = None
        self.guard_command = None
        self.target_selection_pair = None
        self.target_selection_prefers_next = False

    def _is_lane_change_command(self, command):
        return command in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT)

    def _lane_change_distances(self, gps_xy, target_point_world, next_target_point_world):
        start_xy = np.asarray(target_point_world[:2], dtype=np.float32)
        end_xy = np.asarray(next_target_point_world[:2], dtype=np.float32)
        route_dir = end_xy - start_xy
        route_len = float(np.linalg.norm(route_dir))
        if route_len <= 1e-6:
            return None, None, None

        ego_vec = np.asarray(gps_xy[:2], dtype=np.float32) - start_xy
        lateral = abs(route_dir[0] * ego_vec[1] - route_dir[1] * ego_vec[0]) / route_len
        longitudinal = float(np.dot(route_dir, ego_vec) / route_len)
        return float(lateral), longitudinal, route_len

    def _clear_target_point_guard(self):
        self.guard_target_point_world = None
        self.guard_next_target_point_world = None
        self.guard_command = None

    def _guard_lane_change_still_active(self, gps_xy):
        if (
            self.guard_target_point_world is None
            or self.guard_next_target_point_world is None
            or self.guard_command is None
        ):
            return False

        lateral_dist, longitudinal_dist, route_len = self._lane_change_distances(
            gps_xy,
            self.guard_target_point_world,
            self.guard_next_target_point_world,
        )
        if lateral_dist is None:
            self._clear_target_point_guard()
            return False

        if longitudinal_dist is not None and route_len is not None and longitudinal_dist >= route_len:
            self._clear_target_point_guard()
            return False

        if lateral_dist <= 2.5:
            self._clear_target_point_guard()
            return False

        return True

    def _apply_target_point_guard(self, gps_xy, target_point, next_target_point, far_command):
        target_world = np.asarray(target_point[:2], dtype=np.float32)
        next_target_world = np.asarray(next_target_point[:2], dtype=np.float32)
        guard_active = self._guard_lane_change_still_active(gps_xy)

        if self._is_lane_change_command(far_command):
            if guard_active:
                target_shift = float(np.linalg.norm(target_world - self.guard_target_point_world))
                if target_shift > 1.0:
                    return (
                        self.guard_target_point_world.copy(),
                        self.guard_next_target_point_world.copy(),
                        self.guard_command,
                    )

            self.guard_target_point_world = target_world.copy()
            self.guard_next_target_point_world = next_target_world.copy()
            self.guard_command = far_command
            return target_world, next_target_world, far_command

        if not guard_active:
            return target_world, next_target_world, far_command

        return (
            self.guard_target_point_world.copy(),
            self.guard_next_target_point_world.copy(),
            self.guard_command,
        )

    def _same_target_selection_pair(self, first_point_world, second_point_world):
        if self.target_selection_pair is None:
            return False

        prev_first, prev_second = self.target_selection_pair
        return (
            np.linalg.norm(first_point_world - prev_first) <= 1e-3
            and np.linalg.norm(second_point_world - prev_second) <= 1e-3
        )

    def _select_target_indices(self, waypoint_route, gps_xy, compass):
        target_idx = 0
        if len(waypoint_route) > 1:
            first_point = np.asarray(waypoint_route[0][0][:2], dtype=np.float32)
            second_point = np.asarray(waypoint_route[1][0][:2], dtype=np.float32)
            ego_xy = np.asarray(gps_xy[:2], dtype=np.float32)
            first_delta = first_point - ego_xy
            first_dist = float(np.linalg.norm(first_delta))
            forward_vec = np.array([np.cos(compass), np.sin(compass)], dtype=np.float32)
            first_longitudinal = float(np.dot(first_delta, forward_vec))

            if not self._same_target_selection_pair(first_point, second_point):
                self.target_selection_pair = (first_point.copy(), second_point.copy())
                self.target_selection_prefers_next = False

            # Once the first remaining route point is clearly behind the ego,
            # don't let the target selection bounce back to it.
            if first_longitudinal < -TARGET_POINT_BEHIND_EGO_EPS_M:
                self.target_selection_prefers_next = True
            elif self.target_selection_prefers_next:
                if first_dist >= TARGET_POINT_DEMOTION_DISTANCE_M:
                    self.target_selection_prefers_next = False
            elif first_dist <= TARGET_POINT_PROMOTION_DISTANCE_M:
                self.target_selection_prefers_next = True

            target_idx = 1 if self.target_selection_prefers_next else 0
        else:
            self.target_selection_pair = None
            self.target_selection_prefers_next = False

        next_idx = min(target_idx + 1, len(waypoint_route) - 1)
        return target_idx, next_idx

    def _maybe_compensate_lidar_pose(self, gps_xy, yaw, speed):
        gps_xy = np.asarray(gps_xy, dtype=np.float64)
        yaw = float(yaw)

        if not LIDAR_LATENCY_COMPENSATION or speed < 0.5:
            return gps_xy, yaw

        # Avoid double-compensating when lidar pose already comes from a compensated localizer.
        if LIDAR_POSE_SOURCE == 'localizer' and LOCALIZER_LATENCY_COMPENSATION:
            return gps_xy, yaw

        predicted_state = bicycle_model_forward(
            np.array([gps_xy[0], gps_xy[1], yaw, speed], dtype=np.float64),
            self.carla_frame_rate,
            self.control.steer,
            self.control.throttle,
            self.control.brake,
        )
        return predicted_state[0:2], float(predicted_state[2])

    def tick(self, input_data):
        """
        Same as base tick(), but replaces the UKF block with EgoLocalizer.
        Only the localization section (lines 752-767 of base) is changed;
        everything else is identical.
        """
        self.step += 1
        rgb_front = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        lidar_ego = lidar_to_ego_coordinate(input_data['LIDAR'])

        gps_full = input_data['GPS'][1]  # [lat, lon, altitude]
        gps_pos = self._route_planner.convert_gps_to_carla(gps_full)

        # Handle compass NaN
        compass_raw = input_data['IMU'][1][-1]
        if math.isnan(compass_raw):
            compass_raw = 0.0

        # Preprocess compass to CARLA coordinate system
        compass = t_u.preprocess_compass(compass_raw)

        # Get speed
        speed = input_data['SPEED'][1]['speed']

        # ---- EgoLocalizer: used for target_point ----
        gps_raw_xy = np.array([gps_pos[0], gps_pos[1]], dtype=np.float64)
        localized_pos, localized_yaw = self.localizer.update(
            gps_xy=gps_raw_xy,
            compass=compass,
            speed=speed,
            steer=self.control.steer,
            throttle=self.control.throttle,
            brake=float(self.control.brake > 0.5),
        )
        gps_filtered = localized_pos.astype(np.float64)
        compass_filtered = localized_yaw

        # ---- UKF: used for lidar alignment (matches training) ----
        if LIDAR_POSE_SOURCE == 'ukf':
            if not self.filter_initialized:
                self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
                self.filter_initialized = True
            self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
            self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
            ukf_state = self.ukf.x
            self.state_log.append(ukf_state)
            gps_lidar = ukf_state[0:2]
            compass_lidar = ukf_state[2]
        else:
            gps_lidar = gps_filtered
            compass_lidar = compass_filtered

        gps_lidar, compass_lidar = self._maybe_compensate_lidar_pose(
            gps_lidar, compass_lidar, speed
        )

        gps_target_pose, compass_target_pose, target_pose_source = self._resolve_target_pose(
            gps_raw=np.array([gps_pos[0], gps_pos[1]], dtype=np.float32),
            gps_filtered=gps_filtered,
            compass_raw=compass,
            compass_filtered=compass_filtered,
        )

        # --- Lidar alignment uses gps_lidar/compass_lidar ---

        # Combine two frames of lidar data
        if self.last_lidar is not None and self.last_ego_transform is not None:
            current_pos = np.array([gps_lidar[0], gps_lidar[1], 0.0])
            last_pos = np.array([self.last_ego_transform['gps'][0], self.last_ego_transform['gps'][1], 0.0])
            relative_translation = current_pos - last_pos

            current_yaw = compass_lidar
            last_yaw = self.last_ego_transform['compass']
            relative_rotation = current_yaw - last_yaw

            rotation_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw), 0.0],
                                        [np.sin(current_yaw), np.cos(current_yaw), 0.0],
                                        [0.0, 0.0, 1.0]])
            relative_translation_local = rotation_matrix.T @ relative_translation

            lidar_last = algin_lidar(self.last_lidar, relative_translation_local, relative_rotation)
            lidar_combined = np.concatenate((lidar_ego, lidar_last), axis=0)
        else:
            lidar_combined = lidar_ego

        self.last_lidar = lidar_ego
        self.last_ego_transform = {'gps': gps_lidar, 'compass': compass_lidar}

        lidar_bev_img = generate_lidar_bev_images(
            np.copy(lidar_combined),
            saving_name=None,
            img_height=448,
            img_width=448
        )
        lidar_bev_tensor = torch.from_numpy(lidar_bev_img).permute(2, 0, 1).float() / 255.0

        # TransFuser processing (must match base agent: JPEG artifacts + crop + transpose)
        transfuser_rgb = input_data['CAM_FRONT'][1][:, :, :3]
        _, compressed_image = cv2.imencode('.jpg', transfuser_rgb)
        transfuser_rgb = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
        transfuser_rgb = cv2.cvtColor(transfuser_rgb, cv2.COLOR_BGR2RGB)
        transfuser_rgb = transfuser_t_u.crop_array(self.transfuser_config, transfuser_rgb)
        transfuser_rgb = np.transpose(transfuser_rgb, (2, 0, 1))
        transfuser_rgb_tensor = torch.from_numpy(transfuser_rgb).float().unsqueeze(0).to('cuda')

        transfuser_lidar = transfuser_t_u.lidar_to_ego_coordinate(self.transfuser_config, input_data['LIDAR'])

        self.transfuser_state_log.append([gps_lidar[0], gps_lidar[1], compass_lidar, speed])

        if self.transfuser_lidar_last is not None and len(self.transfuser_state_log) >= 2:
            ego_x = self.transfuser_state_log[-1][0]
            ego_y = self.transfuser_state_log[-1][1]
            ego_theta = self.transfuser_state_log[-1][2]

            ego_x_last = self.transfuser_state_log[-2][0]
            ego_y_last = self.transfuser_state_log[-2][1]
            ego_theta_last = self.transfuser_state_log[-2][2]

            transfuser_lidar_last_aligned = self._align_lidar_transfuser(
                self.transfuser_lidar_last,
                ego_x_last, ego_y_last, ego_theta_last,
                ego_x, ego_y, ego_theta
            )
            transfuser_lidar_full = np.concatenate((transfuser_lidar, transfuser_lidar_last_aligned), axis=0)
        else:
            transfuser_lidar_full = transfuser_lidar

        self.transfuser_lidar_last = transfuser_lidar.copy()
        self.transfuser_lidar_buffer.append(transfuser_lidar_full)

        transfuser_lidar_bev = self.transfuser_data.lidar_to_histogram_features(
            transfuser_lidar_full,
            use_ground_plane=self.transfuser_config.use_ground_plane
        )
        transfuser_lidar_bev_tensor = torch.from_numpy(transfuser_lidar_bev).float().unsqueeze(0).to('cuda')

        if IS_BENCH2DRIVE:
            bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            bev = np.zeros((512, 512, 3), dtype=np.uint8)

        result = {
            'rgb_front': rgb_front,
            'lidar_bev': lidar_bev_tensor,
            'gps': gps_target_pose,
            'gps_raw': np.array([gps_pos[0], gps_pos[1]], dtype=np.float32),
            'speed': speed,
            'compass': compass_target_pose,
            'compass_raw': compass,
            'bev': bev,
            'gps_filtered': gps_filtered,
            'compass_filtered': compass_filtered,
            'target_pose_source': target_pose_source,
            'transfuser_rgb': transfuser_rgb_tensor,
            'transfuser_lidar_bev': transfuser_lidar_bev_tensor,
        }

        waypoint_route = self._route_planner.run_step(np.append(result['gps'], gps_pos[2]))

        if len(waypoint_route) > 1:
            target_idx, next_idx = self._select_target_indices(
                waypoint_route,
                result['gps'],
                result['compass'],
            )
            target_point, far_command = waypoint_route[target_idx]
            if next_idx > target_idx:
                next_target_point, next_far_command = waypoint_route[next_idx]
            else:
                ego_pos = result['gps'][:2]
                direction = target_point[:2] - ego_pos
                dist = np.linalg.norm(direction)
                if dist > 1e-3:
                    direction_normalized = direction / dist
                else:
                    direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
                next_target_point = target_point[:2] + direction_normalized * 50.0
                next_far_command = far_command
        elif len(waypoint_route) > 0:
            target_point, far_command = waypoint_route[0]
            ego_pos = result['gps'][:2]
            direction = target_point[:2] - ego_pos
            dist = np.linalg.norm(direction)
            if dist > 1e-3:
                direction_normalized = direction / dist
            else:
                direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
            next_target_point = target_point[:2] + direction_normalized * 50.0
            next_far_command = far_command
        else:
            target_point, far_command = (result['gps'][:2], RoadOption.LANEFOLLOW)
            direction_normalized = np.array([np.cos(result['compass']), np.sin(result['compass'])])
            next_target_point = result['gps'][:2] + direction_normalized * 50.0
            next_far_command = RoadOption.LANEFOLLOW

        target_point, next_target_point, far_command = self._apply_target_point_guard(
            result['gps'],
            target_point,
            next_target_point,
            far_command,
        )

        if self.last_command_tmp != far_command:
            self.last_command = self.last_command_tmp
        self.last_command_tmp = far_command

        if hasattr(target_point, '__iter__') and len(target_point) >= 2:
            if (target_point[:2] != self.target_point_prev[:2]).any() if isinstance(target_point, np.ndarray) else (list(target_point[:2]) != list(self.target_point_prev[:2])):
                self.target_point_prev = target_point
                self.commands.append(far_command.value)

        result['next_command'] = self.commands[-2]
        ego_waypoint_route = []
        for route_item in waypoint_route:
            route_point = route_item[0]
            ego_route_point = t_u.inverse_conversion_2d(route_point[:2], result['gps'], result['compass'])
            ego_waypoint_route.append(ego_route_point)
        if ego_waypoint_route:
            self.last_waypoint_route = np.asarray(ego_waypoint_route, dtype=np.float32)
        else:
            self.last_waypoint_route = None

        ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])
        ego_next_target_point = t_u.inverse_conversion_2d(next_target_point[:2], result['gps'], result['compass'])

        forward_vec_world = np.array([
            np.cos(result['compass']),
            np.sin(result['compass']),
        ], dtype=np.float32)
        target_delta_world = np.asarray(target_point[:2], dtype=np.float32) - np.asarray(result['gps'][:2], dtype=np.float32)
        next_target_delta_world = np.asarray(next_target_point[:2], dtype=np.float32) - np.asarray(result['gps'][:2], dtype=np.float32)

        def _angle_and_dot(delta_world):
            dist = float(np.linalg.norm(delta_world))
            if dist < 1e-6:
                return 0.0, 0.0, 0.0
            dot = float(np.dot(forward_vec_world, delta_world))
            cross = float(forward_vec_world[0] * delta_world[1] - forward_vec_world[1] * delta_world[0])
            cos_val = float(np.clip(dot / dist, -1.0, 1.0))
            angle_deg = float(np.rad2deg(np.arccos(cos_val)))
            return angle_deg, dot, cross

        target_angle_deg, target_dot_forward, target_cross_forward = _angle_and_dot(target_delta_world)
        next_target_angle_deg, next_target_dot_forward, next_target_cross_forward = _angle_and_dot(next_target_delta_world)

        result['target_point'] = ego_target_point
        result['next_target_point'] = ego_next_target_point
        result['target_point_world'] = np.asarray(target_point[:2], dtype=np.float32)
        result['next_target_point_world'] = np.asarray(next_target_point[:2], dtype=np.float32)
        result['target_dot_forward'] = target_dot_forward
        result['next_target_dot_forward'] = next_target_dot_forward
        result['target_cross_forward'] = target_cross_forward
        result['next_target_cross_forward'] = next_target_cross_forward
        result['target_angle_deg'] = target_angle_deg
        result['next_target_angle_deg'] = next_target_angle_deg
        result['target_is_behind'] = bool(target_dot_forward < 0.0)
        result['next_target_is_behind'] = bool(next_target_dot_forward < 0.0)
        result['target_is_right'] = bool(target_cross_forward > 0.0)
        result['next_target_is_right'] = bool(next_target_cross_forward > 0.0)
        result['target_is_left'] = bool(target_cross_forward < 0.0)
        result['next_target_is_left'] = bool(next_target_cross_forward < 0.0)
        if len(waypoint_route) > 0:
            result['waypoint_route_world'] = np.asarray([route_item[0][:2] for route_item in waypoint_route], dtype=np.float32)
            result['waypoint_route_ego'] = self.last_waypoint_route.copy() if self.last_waypoint_route is not None else None
        else:
            result['waypoint_route_world'] = None
            result['waypoint_route_ego'] = None
        result['theta'] = compass_filtered

        return result
