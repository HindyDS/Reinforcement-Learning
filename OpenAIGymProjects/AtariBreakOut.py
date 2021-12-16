# 1) Import Dependencies
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

# 2) Test Enviroment
# Download and extract files from http://www.atarimania.com/roms/Roms.rar to working directory
# execute command line: python -m atari_py.import_roms .\ROMS

env_name = "Breakout-v0"
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

# 3) Vectorise Environment and Traing Model
env = make_atari_env("Breakout-v0", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

log_path = os.path.join("Training", "Logs")
model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)

# 3) Save and Reload Model
a2c_path = os.path.join("Training", "Saved Models", "A2C_Breakout_Model")
model.save(a2c_path)

# del model
# model = A2C.load(a2c_path, env)

# 4) Evaluate and Test
env = make_atari_env("Breakout-v0", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
