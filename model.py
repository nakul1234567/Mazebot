from stable_baselines3 import DQN
import gymnasium
import coverage_gridworld
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments for training
vec_env = make_vec_env("sneaky_enemies", n_envs=4)

model = DQN("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("gridworld")

# Single environment for testing
env = gymnasium.make("sneaky_enemies", render_mode="human", predefined_map_list=None, activate_game_status=True)
model = DQN.load("gridworld")
obs, _ = env.reset()
terminated = False
total_reward = 0
while not terminated:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    total_reward += rewards
print(f"Total reward: {total_reward}")
env.close()
