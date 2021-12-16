# 1) Import Dependencies
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

# 2) Test Environment
env_name = "CarRacing-v0"
env = gym.make(env_name)

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

# 3) Train Model
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

log_path = os.path.join("Training", "Logs")
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=100000)

# 4) Save Model
ppo_path = os.path.join("Training", "Saved Models", "PPO_CarRacing_Model")
model.save(ppo_path)

# del model
# model = PPO.load(ppo_path, env)

# 4) Evaluate and Test
evaluate_policy(model, env, n_eval_episodes=10, render=True)