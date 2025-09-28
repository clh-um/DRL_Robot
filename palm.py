import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
from scipy.ndimage import zoom # Import the zoom function for upscaling

class OilPalmPlantationEnv(gym.Env):
    """
    Custom PyBullet Gym environment for simulating an oil palm plantation.

    The environment consists of:
    - A flat ground plane.
    - Rows of cylindrical objects representing oil palm trees.
    - Trenches running alongside the rows of trees.
    - Randomly scattered mud patches/potholes with distinct physical properties.
    """
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode='human'):
        super(OilPalmPlantationEnv, self).__init__()
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # --- Environment Parameters ---
        self.plane_size = 50  # Size of the ground plane (meters)

        # Bumpiness parameters
        self.hf_rows = 256
        self.hf_cols = 256
        self.height_perturbation = 0.2 # Max height of bumps in meters
        self.fine_grain_noise_factor = 0.2 # Proportion of fine-grained noise

        # Tree parameters
        self.tree_rows = 10
        self.trees_per_row = 10
        self.row_spacing = 8.0  # Distance between rows
        self.tree_spacing = 3.0  # Distance between trees in a row
        self.tree_radius = 0.2
        self.tree_height = 8.0
        self.tree_mass = 0 # Static objects

        # Trench parameters
        self.trench_width = 0.5
        self.trench_depth = 0.3 # Increased depth for visibility
        self.trench_length = self.trees_per_row * self.tree_spacing

        # Mud/Pothole parameters
        self.num_mud_patches = 50
        self.mud_patch_radius_range = (0.5, 1.5)
        self.mud_patch_friction = 0.1
        self.mud_patch_rolling_friction = 1.0
        self.mud_patch_contact_stiffness = 1000.0
        self.mud_patch_contact_damping = 10.0
        
        # Define action and observation spaces
        # Action space: [left_wheel_velocity, right_wheel_velocity]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [x, y, z, qx, qy, qz, qw, vx, vy, vz, avx, avy, avz]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        self.robot_id = None
        self.plane_id = None
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


    def _get_terrain_height(self, x, y):
        """Helper function to get terrain height at a given x, y coordinate."""
        # Map world coordinates to heightfield indices
        col = int(((x + self.plane_size / 2) / self.plane_size) * (self.hf_cols - 1))
        row = int(((y + self.plane_size / 2) / self.plane_size) * (self.hf_rows - 1))

        # Clamp indices to be within bounds
        col = max(0, min(col, self.hf_cols - 1))
        row = max(0, min(row, self.hf_rows - 1))

        return self.heightfield_data[row, col]

    def _create_world(self):
        """Creates the simulation world."""
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        
        # --- Create Composite Bumpy Ground Plane ---
        # 1. Create a low-resolution random heightfield for large bumps
        low_res_rows, low_res_cols = 32, 32
        low_res_heightfield = np.random.uniform(-self.height_perturbation, self.height_perturbation, (low_res_rows, low_res_cols))
        
        # 2. Upscale it to the full resolution using cubic interpolation for smoothness
        scale_factor_rows = self.hf_rows / low_res_rows
        scale_factor_cols = self.hf_cols / low_res_cols
        smooth_terrain = zoom(low_res_heightfield, (scale_factor_rows, scale_factor_cols), order=3)
        
        # 3. Create high-frequency noise for fine-grained detail
        fine_grained_noise = np.random.uniform(-self.height_perturbation * self.fine_grain_noise_factor, 
                                               self.height_perturbation * self.fine_grain_noise_factor, 
                                               (self.hf_rows, self.hf_cols))
        
        # 4. Combine both terrains
        self.heightfield_data = (smooth_terrain + fine_grained_noise).astype(np.float32)

        # --- Carve Trenches into the Heightfield Data ---
        start_x_trees = - (self.tree_rows / 2) * self.row_spacing
        start_y_trees = - (self.trees_per_row / 2) * self.tree_spacing

        for i in range(self.tree_rows):
            # Calculate world coordinates for the two trenches in this row
            x_pos1 = start_x_trees + i * self.row_spacing - self.row_spacing/4
            x_pos2 = start_x_trees + i * self.row_spacing + self.row_spacing/4
            y_pos = start_y_trees + self.trench_length/2 - self.tree_spacing/2
            
            for x_center in [x_pos1, x_pos2]:
                # Convert world coordinates of trench boundaries to heightfield indices
                x_min_world, x_max_world = x_center - self.trench_width / 2, x_center + self.trench_width / 2
                y_min_world, y_max_world = y_pos - self.trench_length / 2, y_pos + self.trench_length / 2

                col_min = max(0, int(((x_min_world + self.plane_size / 2) / self.plane_size) * (self.hf_cols - 1)))
                col_max = min(self.hf_cols - 1, int(((x_max_world + self.plane_size / 2) / self.plane_size) * (self.hf_cols - 1)))
                row_min = max(0, int(((y_min_world + self.plane_size / 2) / self.plane_size) * (self.hf_rows - 1)))
                row_max = min(self.hf_rows - 1, int(((y_max_world + self.plane_size / 2) / self.plane_size) * (self.hf_rows - 1)))
                
                # Lower the height values in the trench area
                self.heightfield_data[row_min:row_max+1, col_min:col_max+1] -= self.trench_depth

        # --- Create the final terrain shape from the modified heightfield data ---
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.plane_size / (self.hf_cols - 1), self.plane_size / (self.hf_rows - 1), 1],
            heightfieldTextureScaling=(self.hf_rows - 1) / 2,
            heightfieldData=self.heightfield_data.flatten(),
            numHeightfieldRows=self.hf_rows,
            numHeightfieldColumns=self.hf_cols,
            physicsClientId=self.client
        )
        
        self.plane_id = p.createMultiBody(0, terrain_shape, physicsClientId=self.client)
        # Load and apply texture from 6ha.jpg
        # Make sure '6ha.jpg' is in the same directory as the script.
        try:
            texture_id = p.loadTexture("terrain/6ha.jpg", physicsClientId=self.client)
            p.changeVisualShape(self.plane_id, -1, textureUniqueId=texture_id, physicsClientId=self.client)
        except p.error:
            print("Warning: Could not load texture '6ha.jpg'. Using default green color.")
            p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.5, 0.8, 0.2, 1.0], physicsClientId=self.client)
        
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.8, physicsClientId=self.client)
        
        # --- Create Trees ---
        tree_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.tree_radius, height=self.tree_height, physicsClientId=self.client)
        tree_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=self.tree_radius, length=self.tree_height, rgbaColor=[0.4, 0.2, 0.0, 1.0], physicsClientId=self.client)
        
        for i in range(self.tree_rows):
            for j in range(self.trees_per_row):
                x_pos = start_x_trees + i * self.row_spacing
                y_pos = start_y_trees + j * self.tree_spacing
                z_pos = self._get_terrain_height(x_pos, y_pos)
                pos = [x_pos, y_pos, z_pos + self.tree_height / 2]
                p.createMultiBody(baseMass=self.tree_mass, baseCollisionShapeIndex=tree_shape, baseVisualShapeIndex=tree_visual, basePosition=pos, physicsClientId=self.client)

        # --- Create Mud Patches ---
        for _ in range(self.num_mud_patches):
            radius = np.random.uniform(self.mud_patch_radius_range[0], self.mud_patch_radius_range[1])
            x_pos = np.random.uniform(-self.plane_size/2, self.plane_size/2)
            y_pos = np.random.uniform(-self.plane_size/2, self.plane_size/2)
            z_pos = self._get_terrain_height(x_pos, y_pos)
            pos = [x_pos, y_pos, z_pos + 0.01] # Slightly above ground
                   
            mud_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=0.02, physicsClientId=self.client)
            mud_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=0.02, rgbaColor=[0.5, 0.3, 0.1, 0.7], physicsClientId=self.client)
            mud_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=mud_shape, baseVisualShapeIndex=mud_visual, basePosition=pos, physicsClientId=self.client)
            
            p.changeDynamics(mud_body, -1, 
                             lateralFriction=self.mud_patch_friction,
                             rollingFriction=self.mud_patch_rolling_friction,
                             contactStiffness=self.mud_patch_contact_stiffness,
                             contactDamping=self.mud_patch_contact_damping,
                             physicsClientId=self.client)

    def _get_observation(self):
        """Gets the robot's observation."""
        if self.robot_id is None:
            return np.zeros(13)
        pos, ori = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        return np.concatenate([pos, ori, vel[0], vel[1]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.client)
        self._create_world()
        
        # Placeholder for robot spawning. Replace with your URDF.
        # For now, we spawn a simple cube.
        robot_start_pos_x = 0
        robot_start_pos_y = 0
        z_pos = self._get_terrain_height(robot_start_pos_x, robot_start_pos_y)
        robot_start_pos = [robot_start_pos_x, robot_start_pos_y, z_pos + 0.5]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("cube.urdf", robot_start_pos, robot_start_orientation, physicsClientId=self.client)

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # This is a placeholder for robot control.
        # You would apply motor commands based on the action.
        # For the cube, we can apply a force.
        p.applyExternalForce(self.robot_id, -1, 
                             forceObj=[action[0]*10, action[1]*10, 0], 
                             posObj=[0,0,0], 
                             flags=p.WORLD_FRAME,
                             physicsClientId=self.client)
        
        p.stepSimulation(physicsClientId=self.client)
        
        observation = self._get_observation()
        
        # Placeholder for reward and termination logic
        reward = 1.0 
        terminated = False
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=10,
                yaw=50,
                pitch=-35,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.client)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self.client)
            (_, _, px, _, _) = p.getCameraImage(
                width=224,
                height=224,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client)
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        p.disconnect(physicsClientId=self.client)

if __name__ == '__main__':
    # --- Example Usage ---
    env = OilPalmPlantationEnv(render_mode='human')
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            
        time.sleep(1./240.) # Simulate in real-time
        
    env.close()

