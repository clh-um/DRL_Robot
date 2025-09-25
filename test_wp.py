from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from robot_env_wp import RobotWaypointEnv  # UPDATED import
import pybullet as p
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import warnings
import matplotlib.image as mpimg
import cloudpickle  # NEW: to load VecNormalize stats

def load_vecnorm_stats(vecnorm_path: str):
    """
    Load VecNormalize stats (obs_rms, clip_obs, etc.) without wrapping the env.
    Returns an object with attributes: obs_rms.mean, obs_rms.var, clip_obs.
    """
    if vecnorm_path and os.path.exists(vecnorm_path):
        try:
            with open(vecnorm_path, "rb") as f:
                vecnorm = cloudpickle.load(f)
            # Basic validation
            _ = vecnorm.obs_rms.mean
            _ = vecnorm.obs_rms.var
            _ = vecnorm.clip_obs
            print(f"Loaded VecNormalize stats from: {vecnorm_path}")
            return vecnorm
        except Exception as e:
            print(f"[WARN] Failed to load VecNormalize stats ({vecnorm_path}): {e}")
    else:
        print("[INFO] VecNormalize stats not found; running without observation normalization.")
    return None

def normalize_obs(obs: np.ndarray, vecnorm) -> np.ndarray:
    """
    Apply VecNormalize-style observation normalization for a single observation.
    """
    if vecnorm is None or vecnorm.obs_rms is None:
        return obs
    eps = 1e-8
    mean = vecnorm.obs_rms.mean
    var = vecnorm.obs_rms.var
    clip = getattr(vecnorm, "clip_obs", 10.0)
    obs_norm = (obs - mean) / np.sqrt(var + eps)
    obs_norm = np.clip(obs_norm, -clip, clip)
    return obs_norm

def evaluate_with_manual_waypoints(
    model_path,
    vecnorm_path=None,       # NEW: path to saved VecNormalize stats (e.g., ./trained_models/vecnormalize_final.pkl)
    num_episodes=2,
    num_waypoints=5,
    render=True
):
    """Evaluate robot performance with manually placed waypoints (GUI)."""

    # Create base environment with manual waypoint placement
    base_env = RobotWaypointEnv(
        render_mode='human' if render else None,
        num_waypoints=num_waypoints,
        manual_waypoints=True  # Enable manual waypoint placement (ensure your env version supports it)
    )

    # Wrap with Monitor (keeps your file logging behavior)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoint_logs")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(base_env, log_dir)

    # Force CPU to avoid the SB3 GPU warning for MLP policies
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the model
    model = PPO.load(model_path, device=device)

    # Load VecNormalize stats for obs normalization (important for performance)
    vecnorm_stats = load_vecnorm_stats(vecnorm_path)

    success_count = 0
    episode_lengths = []
    episode_rewards = []
    trajectories = []
    waypoint_positions = []

    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}: Place waypoints in the pop-up window")
        obs, info = env.reset()  # This will open the waypoint placement UI if supported

        # Store waypoint positions for plotting
        waypoints = [(wp[0], wp[1]) for wp in env.env.waypoints]  # env.env -> underlying RobotWaypointEnv

        episode_reward = 0.0
        episode_length = 0
        trajectory = []

        # Initial position
        robot_pos, _ = p.getBasePositionAndOrientation(env.env.robot_id)
        trajectory.append((robot_pos[0], robot_pos[1]))

        terminated = False
        truncated = False
        print(f"Starting episode with {len(waypoints)} waypoints")

        while not (terminated or truncated):
            # Normalize observation to match training distribution
            obs_in = normalize_obs(obs, vecnorm_stats)

            action, _ = model.predict(obs_in, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Record trajectory
            robot_pos, _ = p.getBasePositionAndOrientation(env.env.robot_id)
            trajectory.append((robot_pos[0], robot_pos[1]))

            episode_reward += reward
            episode_length += 1

            if render:
                time.sleep(0.01)

        success = bool(terminated) and not bool(truncated)
        success_count += int(success)
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        trajectories.append(trajectory)
        waypoint_positions.append(waypoints)

        print(f"Episode {episode+1} finished: {'Success' if success else 'Failure'} | "
              f"Reward: {episode_reward:.2f}")

        if render:
            time.sleep(1)

    env.close()

    success_rate = success_count / max(1, num_episodes)
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0

    print("\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"Average Episode Reward: {avg_reward:.2f}")

    plot_waypoint_trajectories(trajectories, waypoint_positions)

    return trajectories, waypoint_positions, success_rate, avg_length, avg_reward

def plot_waypoint_trajectories(trajectories, waypoint_positions):
    plt.figure(figsize=(12, 10))
    for i, (trajectory, waypoints) in enumerate(zip(trajectories, waypoint_positions)):
        # Trajectory
        traj_x, traj_y = zip(*trajectory)
        plt.plot(traj_x, traj_y, '-', linewidth=1, alpha=0.7, label=f'Episode {i+1}')
        plt.scatter(traj_x[0], traj_y[0], marker='o', color='green', s=100, label='Start' if i == 0 else "")
        plt.scatter(traj_x[-1], traj_y[-1], marker='x', color='blue', s=100, label='End' if i == 0 else "")

        # Waypoints
        if waypoints:
            wp_x, wp_y = zip(*waypoints)
            plt.scatter(wp_x, wp_y, marker='*', color=f'C{i}', s=150)
            plt.plot(wp_x, wp_y, 'k:', alpha=0.5)
            for j, (x, y) in enumerate(waypoints):
                plt.text(x, y, f"{j+1}", fontsize=12, ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='circle'))

    plt.title('Robot Waypoint Navigation Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.savefig('manual_waypoint_trajectories.png')
    plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*Monitor wrapper.*")

    # Adjust paths to your training outputs:
    model_path = os.path.join("trained_models", "ppo_nav_final.zip")  # UPDATED default to match training script
    vecnorm_path = os.path.join("trained_models", "vecnormalize_final.pkl")  # NEW: VecNormalize stats

    print("Running evaluation with manual waypoint placement...")
    trajectories, waypoint_positions, success_rate, avg_length, avg_reward = evaluate_with_manual_waypoints(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        num_episodes=2,
        num_waypoints=5,
        render=True
    )