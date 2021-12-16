# 1) Import Dependencies
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# 2) Building an Environment
class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        # Apply Temperature Adjust
        self.state += (action) - 1

        # Decrease Shower Time
        self.shower_length -= 1

        # Calculate Reward
        if (self.state >= 37) and (self.state <= 39):
            reward = 1

        else:
            reward = -1

        if self.shower_length <= 0:
            done = True

        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def render(self):
        # Implement Viz
        pass

    def reset(self):
        self.state = np.array([38 + random.randint(-3, 3)]).astype(float)
        self.shower_length = 60

        return self.state

# 3) Test Enviroment
env = ShowerEnv()

episodes = 5
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        score += reward
    print(f"Episode:{episode}, Score:{score}")
env.close()

# 4) Train Model
log_path = os.path.join("Training", "Logs")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=100000)

# 4) Save Model
ppo_path = os.path.join("Training", "Saved Models", "PPO_CustomEnviroment_Model")
model.save(ppo_path)

# del model
# model = PPO.load(ppo_path, env)

# 5) Evaluate and Test
evaluate_policy(model, env, n_eval_episodes=10, render=True)