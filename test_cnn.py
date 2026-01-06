from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=4, seed=0)
# Frame-stacking with 6 frames
env = VecFrameStack(env, n_stack=6)
obs = env.reset()

print(obs.shape)
model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=25000)


