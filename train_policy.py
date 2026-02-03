import argparse
import time
import threading
#
import os
from os.path import join, exists
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from flightgym import AvoidVisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from mav_baselines.torch.recurrent_ppo.policies import MultiInputLstmPolicy, CnnLstmPolicy
from mav_baselines.torch.recurrent_ppo.ppo_recurrent import RecurrentPPO

from mav_baselines.torch.envs import vec_multi_env_wrapper as wrapper
from mav_baselines.torch.common.util import test_vision_policy
sys.modules['rpg_baselines_prev'] = mav_baselines
unity_ready = False
save_finished = False
def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def rendering_thread(env):
  global unity_ready, save_finished
  time.sleep(0.1)
  while(True):
    if(unity_ready):
      env.render(0)
      time.sleep(0.01)
      if save_finished:
        break

def learning_rate_schedule(progress_remaining):
    """
    Custom learning rate schedule.
    :param progress_remaining: A float, the proportion of training remaining (1 at the beginning, 0 at the end)
    :return: The learning rate as a float.
    """
    # Example: Linearly decreasing learning rate
    return 1e-4 * progress_remaining

def parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
  parser.add_argument("--render", type=int, default=1, help="Render with Unity")
  parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
  parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
  parser.add_argument("--retrain", type=int, default=0, help="if retrain")
  parser.add_argument("--scene_id", type=int, default=0, help="indoor")
  parser.add_argument("--nocontrol", type=int, default=0, help="if load action and value net parameters")
  parser.add_argument("--rollouts", type=int, default=1000, help="Number of rollouts")
  parser.add_argument("--dir", type=str, default="./datasets",
                      help="Where to place rollouts")
  parser.add_argument('--logdir', type=str, default="./exp_dir",
                      help='Directory where results are logged')
  return parser

def main():
  args = parser().parse_args()
  #1. 配置加载阶段 load configurations
  if args.scene_id == 0:
    cfg = YAML().load(
        open(
            os.environ["AVOIDBENCH_PATH"] + "/../mavrl/configs/control/config_new.yaml", "r"
        )
    )
  else:
    cfg = YAML().load(
        open(
            os.environ["AVOIDBENCH_PATH"] + "/../mavrl/configs/control/config_new_out.yaml", "r"
        )
    )
  #2. 环境初始化阶段
  """
  AvoidVisionEnv_v1 是 flightgym 的 Unity 环境接口。
  VisionEnvVec 是项目的包装器，把环境变成 SB3 可用的 VecEnv 格式。
  """
  train_env = AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
  train_env = wrapper.VisionEnvVec(train_env, logdir=args.logdir)
  # set random seed
  configure_random_seed(args.seed, env=train_env)

  #3. 评估环境创建
  old_num_envs = cfg["simulation"]["num_envs"]
  old_render = cfg["unity"]["render"]#原先配置
  cfg["simulation"]["num_envs"] = 1
  cfg["unity"]["render"] = "no"
  eval_env = wrapper.VisionEnvVec(#创建评估环境
      AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False), logdir=args.logdir
  )
  
  cfg["simulation"]["num_envs"] = old_num_envs
  cfg["unity"]["render"] = old_render
  #关键步骤：让评估环境 共享训练环境的Unity连接 避免重复建立Unity连接，提高效率
  eval_env.wrapper.setUnityFromPtr(train_env.wrapper.getUnityPtr())

  # eval_env.getPointClouds('', 0, False)
  # save the configuration and other files
  rsg_root = os.path.dirname(os.path.abspath(__file__))
  log_dir = rsg_root + "/saved"
  #4. 异步渲染线程
  new_thread = threading.Thread(target=rendering_thread, args=(train_env,))
  new_thread.start()
  """
  5. Unity仿真连接
  异步渲染：通过独立线程处理可视化，不影响训练效率
  Unity环境准备：建立连接、生成障碍物、处理点云数据
  环境同步：确保训练和评估环境使用相同的数据
  状态管理：通过全局变量协调多线程间的操作顺序
  """
  device = get_device("auto")
  if args.render:
    global unity_ready, save_finished
    unity_ready = train_env.connectUnity()
    train_env.spawnObstacles(change_obs=True)#生成障碍物
    while not train_env.ifSceneChanged():
      train_env.spawnObstacles(change_obs=False)
      time.sleep(0.01)
    train_env.getPointClouds('', 0, True)
    while(not train_env.getSavingState()):
      time.sleep(0.02)
    time.sleep(5.0)
    train_env.readPointClouds(0)
    while(not train_env.getReadingState()):#点云数据是由Unity仿真环境根据场景中的障碍物和深度图生成的虚拟点云
      time.sleep(0.02)
    time.sleep(1.0)
    eval_env.readPointClouds(0)
    while(not eval_env.getReadingState()):
      time.sleep(0.02)
    time.sleep(1.0)
    save_finished = True    
  
  """
  6. 模型策略加载与训练
  条件分支：
  （1）重训练/测试：从检查点加载模型
  （2）新训练：创建新策略网络

  模型恢复：支持从检查点继续训练或测试
  灵活加载：可以选择性加载网络参数
  设备适配：自动适应可用的硬件设备
  网络配置：根据需要调整网络结构  
  """
  #当需要重训练(args.retrain=1)或测试模式(args.train=0)时执行
  if (args.retrain or not args.train):
    #模型权重加载
    weight = os.environ["AVOIDBENCH_PATH"] + "/../mavrl/saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
    saved_variables = torch.load(weight, map_location=device)
    # 策略网络创建,     创建LSTM策略网络，特征维度为64,使用保存的网络结构参数初始化
    policy = MultiInputLstmPolicy(features_dim=64, **saved_variables["data"])
    #动作网络调整,      在动作网络末尾添加Tanh激活函数，限制输出范围
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    # 标准差设置， 设置动作分布的标准差参数
    saved_variables["state_dict"]['log_std'] = torch.tensor([-0.0, -0.0, -0.0, -0.0], device=device)
    #选择性参数加载控制
    #当args.nocontrol=1时，移除动作网络和价值网络的参数；这样只加载基础网络结构，而不加载控制相关的参数
    if args.nocontrol:
      saved_variables["state_dict"].pop('action_net.0.weight')
      saved_variables["state_dict"].pop('action_net.0.bias')
      saved_variables["state_dict"].pop('value_net.weight')
      saved_variables["state_dict"].pop('value_net.bias')
      saved_variables["state_dict"].pop('mlp_extractor.value_net.0.weight')
      saved_variables["state_dict"].pop('mlp_extractor.value_net.0.bias')
      saved_variables["state_dict"].pop('mlp_extractor.policy_net.0.weight')
      saved_variables["state_dict"].pop('mlp_extractor.policy_net.0.bias')
      saved_variables["state_dict"].pop('mlp_extractor.policy_net.2.weight')
      saved_variables["state_dict"].pop('mlp_extractor.policy_net.2.bias')
      saved_variables["state_dict"].pop('mlp_extractor.value_net.2.weight')
      saved_variables["state_dict"].pop('mlp_extractor.value_net.2.bias')
    #权重加载
    policy.load_state_dict(saved_variables["state_dict"], strict=False)
    # policy.log_std_init = -0.5
    policy.to(device)
  else:#新训练模式
    #当不需要加载预训练模型时，直接使用策略名称字符串
    policy = "MultiInputLstmPolicy"

  """
  7. PPO训练模型参数构建

  算法选择：使用递归PPO（RecurrentPPO）算法
  网络架构：策略网络[256,256]，价值网络[512,512]
  超参数配置：学习率、折扣因子、批次大小等
  """
  if args.train:
    model = RecurrentPPO(
      tensorboard_log=log_dir,
      policy=policy,
      policy_kwargs=dict(
          activation_fn=torch.nn.ReLU,
          net_arch=[dict(pi=[256, 256], vf=[512, 512])],
          log_std_init=-0.5,
          use_beta = False,
      ),#初始化后会在 _setup_model() 里创建 policy 配置策略网络（Actor/Critic）的结构与行为。
      env=train_env,
      learning_rate=learning_rate_schedule,
      eval_env=eval_env,
      use_tanh_act=True,
      gae_lambda=0.95,
      gamma=0.99,
      n_steps=1000,
      n_seq=1,
      ent_coef=0.0,
      vf_coef=0.2,
      max_grad_norm=0.5,
      lstm_layer=1,
      batch_size=4000,
      clip_range=0.2,
      use_sde=False,  # don't use (gSDE), doesn't work
      retrain=args.retrain,
      device=device,
      env_cfg=cfg,
      verbose=1,
      states_dim=0,
      features_dim=64,
      if_change_maps=True,
      is_forest_env=(args.scene_id==1),
    )
    #8. 训练/测试执行。
    #训练模式：执行800万时间步的强化学习训练
    model.learn(total_timesteps=int(8e6), log_interval=(10, 20))
    
    #测试模式：在评估环境上测试策略性能
  else:
    test_vision_policy(eval_env, policy)

if __name__ == "__main__":
  main()

