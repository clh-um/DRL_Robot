import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import math
import time
import sys
import random

script_path = os.path.dirname(os.path.abspath(__file__))
if script_path not in sys.path:
    sys.path.insert(0, script_path)

# Import your terrain generator
import terrain_generator as tg


class RobotWaypointEnv(gym.Env):
    """
    Differential drive waypoint navigation environment with improved turning,
    reward clarity, curriculum, and stuck detection.

    ACTION (continuous):
        action[0] = normalized linear velocity command  v_cmd in [-1, 1]
        action[1] = normalized angular velocity command w_cmd in [-1, 1]

    OBSERVATION (default, length=7 or extended if history enabled):
        0: distance_to_current_waypoint
        1: angle_diff ([-pi, pi])
        2: planar_linear_speed
        3: yaw_rate (estimated)
        4: last_v_cmd
        5: last_w_cmd
        6: waypoint_progress = current_waypoint_index / total_waypoints
        + (optional history features if enable_history=True)

    Key Features:
      - Forward gating until heading aligns.
      - Optional curriculum (radius shrinking).
      - Stuck detection & early truncation.
      - Reward decomposition returned in info.
      - Stabilized progress reward.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 render_mode=None,
                 num_waypoints=4,
                 manual_waypoints=False,
                 # NEW: domain randomization options
                 vel_noise_std=0.0,
                 heading_noise_std=0.0,
                 # NEW: curriculum
                 enable_curriculum=True,
                 curriculum_min_radius=0.35,
                 curriculum_decay=0.90,
                 # NEW: history augmentation
                 enable_history=False,
                 history_len=3,
                 # NEW: reward shaping toggles
                 reward_scale=1.0,
                 allow_backward=False,
                 max_episode_steps=3000,
                 # Path geometry
                 waypoint_min_dist=3.0,
                 waypoint_max_dist=6.0,
                 prefer_turning=False,
                 seed=None):
        super().__init__()

        self.render_mode = render_mode
        self.manual_waypoints = manual_waypoints
        self.num_waypoints = num_waypoints
        self.cached_wheel_joints = None
        self._cached_robot_id = None
        self._warned_no_wheels = False

        # Random seeds
        self.seed(seed)

        # PyBullet client
        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.resetDebugVisualizerCamera(cameraDistance=4.0,
                                         cameraYaw=45,
                                         cameraPitch=-35,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)

        # Action & observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.enable_history = enable_history
        self.history_len = history_len
        self.history_buffer = []

        base_obs_len = 7
        hist_extra = 0
        if self.enable_history:
            # store (dist, angle) pairs (2 each) for past history_len steps
            hist_extra = 2 * self.history_len

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(base_obs_len + hist_extra,),
                                            dtype=np.float32)

        # Paths
        self.urdf_path = os.path.join(script_path, "robot_model", "robot.urdf")

        # Kinematics parameters
        self.max_linear_velocity = 2.0
        self.max_angular_velocity = 4.0
        self.wheel_base = 0.40
        self.wheel_radius = 1.0

        # Episode control
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Waypoints
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_visual_ids = []
        self.waypoint_text_ids = []
        self.base_waypoint_radius = 1.0
        self.waypoint_radius = self.base_waypoint_radius

        # Curriculum
        self.enable_curriculum = enable_curriculum
        self.curriculum_min_radius = curriculum_min_radius
        self.curriculum_decay = curriculum_decay
        self.success_counter = 0

        # Generation geometry
        self.waypoint_min_dist = waypoint_min_dist
        self.waypoint_max_dist = waypoint_max_dist
        self.prefer_turning = prefer_turning

        # Domain randomization
        self.vel_noise_std = vel_noise_std
        self.heading_noise_std = heading_noise_std

        # Reward shaping config
        self.reward_scale = reward_scale
        self.allow_backward = allow_backward

        # Reward gains
        self.progress_gain = 8.0
        self.angle_penalty_gain = 1.4
        self.angle_small_bonus = 0.6
        self.angle_align_threshold = 0.15
        self.angle_allow_forward = 0.55
        self.forward_misaligned_penalty = 1.1
        self.angular_direction_bonus = 0.55
        self.distance_bonus_gain = 0.35
        self.speed_bonus_gain = 0.05
        self.inactive_penalty = 0.02
        self.reached_waypoint_reward = 7.5
        self.success_terminal_reward = 35.0
        self.fail_terminal_penalty = -5.0
        self.backward_penalty_gain = 0.3
        self.divergence_penalty_gain = 2.0

        # Stuck detection
        self.stuck_distance_delta_threshold = 0.02
        self.stuck_window = 80
        self.recent_progress = []

        # Divergence detection
        self.divergence_window = 40
        self.divergence_counter = 0

        # Internals
        self.prev_distance = None
        self.last_v_cmd = 0.0
        self.last_w_cmd = 0.0
        self.estimated_yaw_rate = 0.0
        self.cached_wheel_joints = None

        self._rng = np.random.default_rng(seed)

    # ---------------- Seeding ----------------
    def seed(self, seed=None):
        if seed is None:
            seed = random.randint(0, 10_000_000)
        np.random.seed(seed)
        random.seed(seed)
        self._seed = seed
        return [seed]

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # Terrain
        texture_path = os.path.join(script_path, "terrain", "6ha.png")
        print(f"Loading texture from: {texture_path}")
        print(f"Texture exists: {os.path.exists(texture_path)}")
        heightfield = tg.generate_procedural_heightfield(rows=100, cols=100, height_perturbation=0.015)
        self.terrain_id = tg.create_pybullet_terrain(heightfield,
                                                     mesh_scale=(3.3, 1.8, 10.0),
                                                     base_position=(0, 0, 0),
                                                     texture_path=texture_path)

        # Random start
        start_x = self._rng.uniform(-2, 2)
        start_y = self._rng.uniform(-2, 2)
        self.robot_id = p.loadURDF(self.urdf_path, [start_x, start_y, 0.2], physicsClientId=self.client)
        self._cache_wheel_joints()

        # Waypoints
        if self.manual_waypoints:
            # Implement manual placement if needed
            self._generate_waypoints(start_x, start_y)  # fallback
        else:
            self._generate_waypoints(start_x, start_y)

        if self.render_mode == 'human':
            self._create_waypoint_visuals()

        self.current_waypoint_idx = 0
        self.current_step = 0
        self.last_v_cmd = 0.0
        self.last_w_cmd = 0.0
        self.history_buffer.clear()
        self.recent_progress = []
        self.divergence_counter = 0

        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        first_distance = np.linalg.norm(self.waypoints[0][:2] - np.array(base_pos)[:2])
        self.prev_distance = first_distance

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        v_cmd_norm, w_cmd_norm = float(action[0]), float(action[1])

        # Optional: disallow backward strongly (clip)
        if not self.allow_backward and v_cmd_norm < 0:
            v_cmd_norm = 0.0

        # Apply action
        self._apply_action(v_cmd_norm, w_cmd_norm)

        # Step physics
        substeps = 10
        yaw_before = self._get_yaw()
        for _ in range(substeps):
            p.stepSimulation(physicsClientId=self.client)
        yaw_after = self._get_yaw()
        dt = substeps / 240.0
        self.estimated_yaw_rate = self._angle_diff(yaw_after, yaw_before) / dt

        # Distances & waypoint logic
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        waypoint = self.waypoints[min(self.current_waypoint_idx, len(self.waypoints)-1)]
        current_distance = np.linalg.norm(waypoint[:2] - np.array(robot_pos)[:2])

        reached_waypoint = current_distance < self.waypoint_radius
        if reached_waypoint:
            self.current_waypoint_idx += 1
            self.success_counter += 1
            if self.enable_curriculum and self.current_waypoint_idx == len(self.waypoints):
                # decay radius for future episodes
                self.base_waypoint_radius = max(self.curriculum_min_radius,
                                                self.base_waypoint_radius * self.curriculum_decay)
            # Update radius for next waypoint
            self.waypoint_radius = self.base_waypoint_radius
            if self.current_waypoint_idx < len(self.waypoints):
                next_wp = self.waypoints[self.current_waypoint_idx]
                self.prev_distance = np.linalg.norm(next_wp[:2] - np.array(robot_pos)[:2])
            if self.render_mode == 'human':
                self._update_waypoint_visuals()

        # Observation
        obs = self._get_observation()

        # Stuck / divergence tracking
        progress_sample = (self.prev_distance - current_distance) if self.prev_distance else 0.0
        self.recent_progress.append(progress_sample)
        if len(self.recent_progress) > self.stuck_window:
            self.recent_progress.pop(0)

        if progress_sample < -0.001:
            self.divergence_counter += 1
        else:
            self.divergence_counter = max(0, self.divergence_counter - 1)

        stuck = self._is_stuck()
        diverging = self.divergence_counter > self.divergence_window

        terminated = self._check_termination()
        truncated = self._check_truncation() or stuck or diverging

        # Reward
        reward, reward_components = self._calculate_reward(
            current_distance=current_distance,
            v_cmd_norm=v_cmd_norm,
            w_cmd_norm=w_cmd_norm,
            reached_waypoint=reached_waypoint,
            terminated=terminated,
            truncated=truncated,
            stuck=stuck,
            diverging=diverging
        )

        # Update prev distance if not just switching
        if not reached_waypoint:
            self.prev_distance = current_distance

        self.current_step += 1

        info = {
            "distance": float(current_distance),
            "current_waypoint": self.current_waypoint_idx,
            "total_waypoints": len(self.waypoints),
            "waypoint_radius": self.waypoint_radius,
            "stuck": stuck,
            "diverging": diverging,
            "reward_components": reward_components
        }

        return obs, reward, bool(terminated), bool(truncated), info

    def render(self):
        pass

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    # ---------------- Kinematics / Actuation ----------------
    def _apply_action(self, v_cmd_norm, w_cmd_norm):
        """
        Safe actuator application:
        - Re-validates wheel joint indices every call.
        - Applies forward gating logic.
        """
        self.last_v_cmd = float(v_cmd_norm)
        self.last_w_cmd = float(w_cmd_norm)

        # Compute heading error for forward gating
        dist, angle_diff = self._distance_and_angle_to_current_wp()
        gating = max(0.0, math.cos(angle_diff))
        if abs(angle_diff) > self.angle_allow_forward:
            gating *= max(0.0, 1.0 - (abs(angle_diff) - self.angle_allow_forward) / (math.pi - self.angle_allow_forward))
        gating = gating ** 1.5

        effective_v = v_cmd_norm * self.max_linear_velocity * gating
        effective_w = w_cmd_norm * self.max_angular_velocity

        v_left = effective_v - (effective_w * self.wheel_base / 2.0)
        v_right = effective_v + (effective_w * self.wheel_base / 2.0)

        speed_limit = self.max_linear_velocity + self.max_angular_velocity * self.wheel_base / 2.0
        v_left = np.clip(v_left, -speed_limit, speed_limit)
        v_right = np.clip(v_right, -speed_limit, speed_limit)

        wheel_joints = self._get_wheel_joints()

        if not wheel_joints:
            if not self._warned_no_wheels:
                print("[WARN] No wheel joints detected for robot_id", self.robot_id,
                    "- skipping actuation this step.")
                self._warned_no_wheels = True
            return

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)

        for jidx, jname in wheel_joints:
            if jidx >= num_joints:
                # Should not happen now, but we guard anyway.
                print(f"[WARN] Skipping invalid joint index {jidx} (num_joints={num_joints}).")
                continue
            if 'left' in jname.lower():
                p.setJointMotorControl2(self.robot_id, jidx, p.VELOCITY_CONTROL,
                                        targetVelocity=v_left, force=30.0,
                                        physicsClientId=self.client)
            elif 'right' in jname.lower():
                p.setJointMotorControl2(self.robot_id, jidx, p.VELOCITY_CONTROL,
                                        targetVelocity=v_right, force=30.0,
                                        physicsClientId=self.client)
            
    def _get_wheel_joints(self):
        """
        Return a valid list of wheel joints. If cache invalid, rebuild.
        """
        # Conditions requiring refresh:
        if (
            self.cached_wheel_joints is None
            or self._cached_robot_id != self.robot_id
        ):
            self._cache_wheel_joints()
        else:
            # Validate indices still in range (paranoia check).
            num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
            invalid = any(jidx >= num_joints for jidx, _ in self.cached_wheel_joints)
            if invalid:
                self._cache_wheel_joints()

        return self.cached_wheel_joints if self.cached_wheel_joints is not None else []

    def _cache_wheel_joints(self):
        """
        (Re)cache wheel joints for the current robot_id safely.
        A wheel joint is identified by 'wheel' substring (case-insensitive) or
        specifically 'wheel_joint' if you want to be stricter.
        """
        if not hasattr(self, 'robot_id'):
            self.cached_wheel_joints = []
            self._cached_robot_id = None
            return

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        wheel_joints = []
        for i in range(num_joints):
            jinfo = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            name = jinfo[1].decode('utf-8')
            lname = name.lower()
            if 'wheel' in lname:  # broaden match in case naming changed
                wheel_joints.append((i, name))

        self.cached_wheel_joints = wheel_joints
        self._cached_robot_id = self.robot_id
        self._warned_no_wheels = False  # reset warning flag

    # ---------------- Observation Helpers ----------------
    def _distance_and_angle_to_current_wp(self):
        robot_pos, robot_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        _, _, yaw = p.getEulerFromQuaternion(robot_quat)

        idx = min(self.current_waypoint_idx, len(self.waypoints)-1)
        wp = self.waypoints[idx]
        vec = wp[:2] - np.array(robot_pos)[:2]
        distance = np.linalg.norm(vec)
        desired_yaw = math.atan2(vec[1], vec[0])
        angle_diff = self._angle_diff(desired_yaw, yaw)

        # Optional heading noise
        if self.heading_noise_std > 0:
            angle_diff += np.random.randn() * self.heading_noise_std
            angle_diff = ((angle_diff + math.pi) % (2 * math.pi)) - math.pi

        return distance, angle_diff

    def _get_yaw(self):
        _, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        return p.getEulerFromQuaternion(quat)[2]

    def _get_observation(self):
        dist, angle_diff = self._distance_and_angle_to_current_wp()
        robot_vel_linear, robot_vel_angular = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        planar_speed = math.sqrt(robot_vel_linear[0] ** 2 + robot_vel_linear[1] ** 2)
        yaw_rate = self.estimated_yaw_rate

        progress = self.current_waypoint_idx / self.num_waypoints if self.num_waypoints > 0 else 0.0

        base_obs = [
            dist,
            angle_diff,
            planar_speed,
            yaw_rate,
            self.last_v_cmd,
            self.last_w_cmd,
            progress
        ]

        if self.enable_history:
            # Push current (dist, angle) into history
            self.history_buffer.append((dist, angle_diff))
            if len(self.history_buffer) > self.history_len:
                self.history_buffer.pop(0)
            # Pad if not full yet
            hist_flat = []
            for d, a in self.history_buffer:
                hist_flat.extend([d, a])
            # pad zeros
            while len(hist_flat) < 2 * self.history_len:
                hist_flat.insert(0, 0.0)
                hist_flat.insert(0, 0.0)
            base_obs.extend(hist_flat)

        return np.array(base_obs, dtype=np.float32)

    # ---------------- Reward ----------------
    def _calculate_reward(self,
                          current_distance,
                          v_cmd_norm,
                          w_cmd_norm,
                          reached_waypoint,
                          terminated,
                          truncated,
                          stuck,
                          diverging):
        dist, angle_diff = self._distance_and_angle_to_current_wp()
        planar_speed = self._get_observation()[2]

        reward_components = {}

        # 1. Progress
        progress = 0.0
        if self.prev_distance is not None and not reached_waypoint:
            raw_prog = self.prev_distance - current_distance
            progress = np.clip(raw_prog, -0.5, 0.5)
        progress_r = self.progress_gain * progress
        reward_components["progress"] = progress_r

        # 2. Angle penalty + alignment bonus
        angle_penalty = -self.angle_penalty_gain * abs(angle_diff)
        small_bonus = self.angle_small_bonus if abs(angle_diff) < self.angle_align_threshold else 0.0
        reward_components["angle_penalty"] = angle_penalty
        reward_components["angle_small_bonus"] = small_bonus

        # 3. Penalize forward when misaligned
        forward_misaligned = 0.0
        if abs(angle_diff) > self.angle_allow_forward and v_cmd_norm > 0.05:
            forward_misaligned = -self.forward_misaligned_penalty * v_cmd_norm * (abs(angle_diff) / math.pi)
        reward_components["forward_misaligned"] = forward_misaligned

        # 4. Angular command direction
        angular_dir_bonus = 0.0
        if abs(angle_diff) > self.angle_align_threshold and w_cmd_norm != 0:
            if np.sign(angle_diff) == np.sign(-w_cmd_norm):
                angular_dir_bonus = self.angular_direction_bonus * min(1.0, abs(angle_diff) / math.pi)
        reward_components["angular_dir_bonus"] = angular_dir_bonus

        # 5. Proximity shaping
        proximity = self.distance_bonus_gain / (1.0 + dist)
        reward_components["proximity"] = proximity

        # 6. Speed usage (only if not severely misaligned)
        speed_bonus = 0.0
        if abs(angle_diff) < self.angle_allow_forward:
            speed_bonus = self.speed_bonus_gain * planar_speed
        reward_components["speed_bonus"] = speed_bonus

        # 7. Inactivity penalty
        inactive_pen = 0.0
        if abs(v_cmd_norm) < 0.05 and abs(w_cmd_norm) < 0.05:
            inactive_pen = -self.inactive_penalty
        reward_components["inactive"] = inactive_pen

        # 8. Backward penalty (if allowed)
        backward_pen = 0.0
        if self.allow_backward and v_cmd_norm < -0.05:
            backward_pen = -self.backward_penalty_gain * abs(v_cmd_norm)
        reward_components["backward_pen"] = backward_pen

        # 9. Waypoint reward
        wp_reward = self.reached_waypoint_reward if reached_waypoint else 0.0
        reward_components["waypoint"] = wp_reward

        # 10. Stuck / divergence penalty
        stuck_pen = -2.0 if stuck else 0.0
        diverge_pen = -self.divergence_penalty_gain if diverging else 0.0
        reward_components["stuck_penalty"] = stuck_pen
        reward_components["diverge_penalty"] = diverge_pen

        # 11. Terminal
        terminal_reward = 0.0
        if terminated:
            terminal_reward = self.success_terminal_reward
        elif truncated and not terminated:
            terminal_reward = self.fail_terminal_penalty
        reward_components["terminal"] = terminal_reward

        total = sum(reward_components.values())
        total *= self.reward_scale

        return total, reward_components

    # ---------------- Termination ----------------
    def _check_termination(self):
        return self.current_waypoint_idx >= len(self.waypoints)

    def _check_truncation(self):
        return self.current_step >= self.max_episode_steps

    def _is_stuck(self):
        # If we have enough samples and almost no positive progress
        if len(self.recent_progress) < self.stuck_window:
            return False
        avg_prog = np.mean(self.recent_progress)
        max_prog = np.max(self.recent_progress)
        return (avg_prog < self.stuck_distance_delta_threshold and
                max_prog < self.stuck_distance_delta_threshold * 1.5)

    # ---------------- Waypoints ----------------
    def _generate_waypoints(self, start_x, start_y):
        self.waypoints.clear()
        for vid in self.waypoint_visual_ids:
            try:
                p.removeBody(vid, physicsClientId=self.client)
            except Exception:
                pass
        self.waypoint_visual_ids.clear()
        self.waypoint_text_ids.clear()

        cx, cy = start_x, start_y
        prev_heading = None
        for i in range(self.num_waypoints):
            dist = self._rng.uniform(self.waypoint_min_dist, self.waypoint_max_dist)

            if self.prefer_turning and prev_heading is not None:
                # Bias heading change
                base_angle = prev_heading + self._rng.uniform(-math.pi/2, math.pi/2)
            else:
                base_angle = self._rng.uniform(0, 2 * math.pi)

            nx = cx + dist * math.cos(base_angle)
            ny = cy + dist * math.sin(base_angle)
            self.waypoints.append(np.array([nx, ny, 0.1]))
            cx, cy = nx, ny
            prev_heading = base_angle

        self.waypoint_radius = self.base_waypoint_radius

    def _create_waypoint_visuals(self):
        for i, wp in enumerate(self.waypoints):
            color = [1, 0, 0, 0.7] if i == self.current_waypoint_idx else [0, 0, 1, 0.7]
            vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color)
            bid = p.createMultiBody(baseVisualShapeIndex=vs, basePosition=wp)
            self.waypoint_visual_ids.append(bid)
            txt = p.addUserDebugText(f"WP{i+1}", wp + np.array([0, 0, 0.3]), [1, 1, 1], 1.5)
            self.waypoint_text_ids.append(txt)

    def _update_waypoint_visuals(self):
        for i in range(len(self.waypoints)):
            try:
                p.removeBody(self.waypoint_visual_ids[i])
            except Exception:
                pass
            if i < self.current_waypoint_idx:
                col = [0, 1, 0, 0.7]
            elif i == self.current_waypoint_idx:
                col = [1, 0, 0, 0.7]
            else:
                col = [0, 0, 1, 0.7]
            vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=col)
            bid = p.createMultiBody(baseVisualShapeIndex=vs, basePosition=self.waypoints[i])
            self.waypoint_visual_ids[i] = bid

    # ---------------- Utility ----------------
    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        return ((d + math.pi) % (2 * math.pi)) - math.pi


# Simple sanity test
if __name__ == "__main__":
    env = RobotWaypointEnv(render_mode='human', num_waypoints=4, enable_curriculum=True)
    obs, _ = env.reset()
    for _ in range(800):
        dist = obs[0]
        angle = obs[1]
        if abs(angle) > 0.3:
            action = np.array([0.2, -np.sign(angle)], dtype=np.float32)
        else:
            action = np.array([0.8, -0.5 * angle], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    env.close()