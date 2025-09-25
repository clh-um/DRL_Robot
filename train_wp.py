import os
import numpy as np
import torch
import gymnasium as gym
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import Logger
from typing import List, Dict, Any
import warnings
import multiprocessing

from robot_env_wp import RobotWaypointEnv  # Make sure filename matches


# =========================
# Custom Callbacks
# =========================

class CurriculumCallback(BaseCallback):
    """
    Monitors evaluation successes and prints current waypoint radius (handled internally in env).
    Useful if you enable curriculum in the environment.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        # You can log aggregated info from vec env
        return True


class RewardDebugCallback(BaseCallback):
    """
    Logs custom reward component means every log_interval steps.
    """
    def __init__(self, log_interval: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.accumulators: Dict[str, float] = {}
        self.count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                for k, v in info["reward_components"].items():
                    self.accumulators.setdefault(k, 0.0)
                    self.accumulators[k] += v
                self.count += 1
        if self.count > 0 and self.n_calls % self.log_interval == 0:
            for k, tot in self.accumulators.items():
                self.logger.record(f"reward_comp/{k}", tot / self.count)
            self.logger.record("reward_comp/count_samples", self.count)
            self.accumulators.clear()
            self.count = 0
        return True


class StuckMonitorCallback(BaseCallback):
    """
    Tracks how many episodes ended due to 'stuck' or 'diverging'.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.stuck_episodes = 0
        self.diverging_episodes = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1
                if info.get("stuck", False):
                    self.stuck_episodes += 1
                if info.get("diverging", False):
                    self.diverging_episodes += 1

        if self.episode_count > 0 and self.n_calls % 5000 == 0:
            self.logger.record("episodes/stuck_rate", self.stuck_episodes / self.episode_count)
            self.logger.record("episodes/diverging_rate", self.diverging_episodes / self.episode_count)
        return True


# =========================
# Environment Factory
# =========================

def make_env(rank: int,
             seed: int,
             num_waypoints: int,
             enable_curriculum: bool,
             prefer_turning: bool):
    """
    Returns an environment creation function for SubprocVecEnv.
    """
    def _init():
        env = RobotWaypointEnv(
            render_mode=None,
            num_waypoints=num_waypoints,
            manual_waypoints=False,
            enable_curriculum=enable_curriculum,
            prefer_turning=prefer_turning,
            # Domain randomization mild:
            vel_noise_std=0.05,
            heading_noise_std=0.01,
            allow_backward=False,
            enable_history=False  # Set True if you want extended obs
        )
        env.reset(seed=seed + rank)
        return env
    return _init


# =========================
# Learning Rate Schedule
# =========================

def linear_schedule(initial_value: float):
    """
    Linear decay schedule for learning rate (and can be used for ent_coef if desired).
    """
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


# =========================
# Training Function
# =========================

def train_robot_waypoint_navigation(
    total_timesteps=1_500_000,
    num_waypoints=4,
    num_cpu=4,
    save_path="./trained_models/",
    log_path="./training_logs/",
    seed=42,
    enable_curriculum=True,
    prefer_turning=True,
    resume=False
):
    multiprocessing.freeze_support()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(f"{save_path}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_path}/best_model", exist_ok=True)

    set_random_seed(seed)

    print("=== Training Configuration ===")
    print(f"Waypoints: {num_waypoints}")
    print(f"CPUs (envs): {num_cpu}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Curriculum Enabled: {enable_curriculum}")
    print(f"Prefer Turning in Path Gen: {prefer_turning}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Vectorized training env
    env_fns = [
        make_env(rank=i,
                 seed=seed,
                 num_waypoints=num_waypoints,
                 enable_curriculum=enable_curriculum,
                 prefer_turning=prefer_turning)
        for i in range(num_cpu)
    ]
    vec_env = SubprocVecEnv(env_fns)

    # IMPORTANT: VecNormalize
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995
    )

    # Evaluation environment (no noise ideally)
    eval_env = DummyVecEnv([
        lambda: RobotWaypointEnv(
            render_mode=None,
            num_waypoints=num_waypoints,
            manual_waypoints=False,
            enable_curriculum=enable_curriculum,
            prefer_turning=prefer_turning,
            vel_noise_std=0.0,
            heading_noise_std=0.0,
            allow_backward=False
        )
    ])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000, total_timesteps // 30),
        save_path=f"{save_path}/checkpoints/",
        name_prefix="ppo_nav",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model/",
        log_path=f"{log_path}/eval/",
        eval_freq=max(20_000, total_timesteps // 50),
        n_eval_episodes=8,
        deterministic=True,
        render=False,
        verbose=1
    )

    reward_debug_cb = RewardDebugCallback(log_interval=10_000)
    stuck_cb = StuckMonitorCallback()
    curriculum_cb = CurriculumCallback()

    callbacks = [checkpoint_callback, eval_callback, reward_debug_cb, stuck_cb, curriculum_cb]

    # PPO Model
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128]
        ),
        activation_fn=torch.nn.ReLU,
        ortho_init=True
    )

    lr_schedule = linear_schedule(3e-4)

    model = None
    latest_checkpoint = None

    if resume:
        # Try to find latest checkpoint
        ckpts = [f for f in os.listdir(f"{save_path}/checkpoints") if f.endswith(".zip")]
        if ckpts:
            ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(f"{save_path}/checkpoints", x)))
            latest_checkpoint = os.path.join(f"{save_path}/checkpoints", ckpts[-1])
            print(f"Resuming from checkpoint: {latest_checkpoint}")

    if latest_checkpoint and resume:
        model = PPO.load(latest_checkpoint, env=vec_env, device=device)
        print("Loaded model from checkpoint.")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=8192,          # Larger batch => more stable gradient for shaped reward
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,        # Slightly lower to focus on exploitation after some steps
            vf_coef=0.55,
            max_grad_norm=0.5,
            tensorboard_log=f"{log_path}/tensorboard/",
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed
        )

    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    print("Training finished.")

    # Save final model + VecNormalize stats
    final_model_path = os.path.join(save_path, "ppo_nav_final")
    model.save(final_model_path)
    vec_env.save(os.path.join(save_path, "vecnormalize_final.pkl"))
    print(f"Saved final model to: {final_model_path}")
    return model, final_model_path, vec_env


# =========================
# Evaluation / Test
# =========================

def load_and_test(model_path: str,
                  vecnorm_path: str,
                  num_episodes: int = 3,
                  num_waypoints: int = 4,
                  render=True):
    print(f"Loading model: {model_path}")
    env = RobotWaypointEnv(
        render_mode='human' if render else None,
        num_waypoints=num_waypoints,
        enable_curriculum=False,
        prefer_turning=True,
        vel_noise_std=0.0,
        heading_noise_std=0.0,
        allow_backward=False
    )
    # Wrap for observation normalization (but do not normalize reward at test unless you prefer)
    test_env = DummyVecEnv([lambda: env])
    if os.path.exists(vecnorm_path):
        vecnorm = VecNormalize.load(vecnorm_path, test_env)
        vecnorm.training = False
        vecnorm.norm_reward = False
        wrapped_env = vecnorm
        print("Loaded VecNormalize stats.")
    else:
        wrapped_env = test_env
        print("VecNormalize stats NOT FOUND; running raw environment.")

    from stable_baselines3 import PPO
    model = PPO.load(model_path, env=wrapped_env)

    successes = 0
    for ep in range(num_episodes):
        obs = wrapped_env.reset()
        done = False
        ep_rew = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = wrapped_env.step(action)
            done = done_vec[0]
            ep_rew += reward[0]
            steps += 1
        info0 = info[0]
        success = info0["current_waypoint"] >= info0["total_waypoints"]
        successes += int(success)
        print(f"Episode {ep+1}: {'SUCCESS' if success else 'FAIL'} | steps={steps} reward={ep_rew:.2f} waypoints={info0['current_waypoint']}/{info0['total_waypoints']}")
    print(f"Success rate: {successes}/{num_episodes}")

    wrapped_env.close()


# =========================
# Main
# =========================

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*Monitor wrapper.*")

    config = {
        "total_timesteps": 1_200_000,
        "num_waypoints": 4,
        "num_cpu": 4,
        "save_path": "./trained_models/",
        "log_path": "./training_logs/",
        "seed": 42,
        "enable_curriculum": True,
        "prefer_turning": True,
        "resume": False
    }

    model, model_path, vec_env = train_robot_waypoint_navigation(**config)

    # Quick test (non-render for speed). To visually test, set render=True below.
    print("\n=== Quick Evaluation ===")
    load_and_test(model_path + ".zip",
                  vecnorm_path=os.path.join(config["save_path"], "vecnormalize_final.pkl"),
                  num_episodes=2,
                  num_waypoints=config["num_waypoints"],
                  render=False)

    print("\nFinished.")