#!/usr/bin/env python3
import argparse
import math
#
import time
import threading

from PIL import Image
import os
from os.path import join, exists
import numpy as np
from flightgym import AvoidVisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device

from mav_baselines.torch.envs import vec_multi_env_wrapper as wrapper
unity_ready = False
save_finished = False

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=0, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    parser.add_argument("--rollouts", type=int, default=1000, help="Number of rollouts")
    parser.add_argument("--dir", type=str, default="./datasets",
                        help="Where to place rollouts")
    return parser

def rendering_thread(env):
  global unity_ready, save_finished
  time.sleep(0.01)
  while True:
    if(unity_ready and not save_finished):
      res = env.render(0)
    time.sleep(0.01)

def change_map(env, seed=-1, radius=-1.0, if_eval=False):
    global save_finished
    save_finished = False
    env.spawnObstacles(change_obs=True, seed=seed, radius=radius)
    while not env.ifSceneChanged():
        env.spawnObstacles(change_obs=False)
        time.sleep(0.01)
    env.getPointClouds('', 0, True)
    time.sleep(1.0)
    while(not env.getSavingState()):
        time.sleep(1.0)
    time.sleep(10.0)
    env.readPointClouds(0)
    while(not env.getReadingState()):
        time.sleep(0.02)
    time.sleep(1.0)
    if not if_eval:
      save_finished = True

def main():
    args = parser().parse_args()
    # load configurations
    cfg = YAML().load(
        open(
            os.environ["AVOIDBENCH_PATH"] + "/../mavrl/configs/control/config_dataset_outdoor.yaml", "r"
        )
    )

    env_ = AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    env_ = wrapper.VisionEnvVec(env_)

    new_thread = threading.Thread(target=rendering_thread, args=(env_,))
    new_thread.start()

    global unity_ready, save_finished
    unity_ready = env_.connectUnity()
    
    generate_data(1000, args.dir, env_)

def generate_data(rollout, data_dir, env):
    for i in range(rollout):
        if i % 10 == 0:
            change_map(env)
        env.reset()
        a_rollout = sample_continuous_policy(env.action_space, 200, env.seq_dim, 0.05)
        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        while True:
            action = a_rollout[t]
            t += 1
            s, r, done, _ = env.step(action)
            depth = env.getDepthImage().reshape(env.img_height, env.img_width)
            depth = (np.minimum(depth, 12.0)) / 12.0 * 255.0
            # depth_int = (np.round(depth)).astype(int)
            depth_img = Image.fromarray(np.uint8(depth))
            # depth_img = depth_img.resize((256, 256))
            # depth_img.save('had'+str(i)+'.jpg')
            depth = np.array(depth_img)
            s_rollout += [depth]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

def sample_continuous_policy(action_space, seq_len, act_len, dt):
    actions = [action_space.sample()[:4]]
    for _ in range(act_len * seq_len-1):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                action_space.low[:4], action_space.high[:4]))
    action_np = np.asarray(actions, dtype=np.float64).reshape(-1,4*act_len)
    return action_np

if __name__ == "__main__":
    main()
