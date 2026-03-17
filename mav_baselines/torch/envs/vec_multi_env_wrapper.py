import os
import pickle
import time
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from PIL import Image
import gym
import numpy as np
from gym import spaces
from numpy.core.fromnumeric import shape
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from stable_baselines3.common.vec_env.util import (copy_obs_dict, dict_to_obs,
                                                   obs_space_info)
from os.path import join, exists
import torch
from utils.misc import LSIZE, n_seq
#负责把底层返回的深度图/状态等数据，整理成 obs 字典（image + state），并维护序列长度 n_seq 的历史帧。
class VisionEnvVec(VecEnv):
    """
    把底层 AvoidVisionEnv_v1 包装成 SB3 的 VecEnv 风格，
    并初始化观测/动作空间、缓存、序列内存。

    train_env = AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.VisionEnvVec(train_env, logdir=args.logdir)
    """
    def __init__(self, impl, logdir=None):
        self.wrapper = impl#impl传进来的是AvoidVisionEnv_v1(dump(cfg, Dumper=RoundTripDumper)
        """
        1) 读底层环境维度
        从 wrapper 读取：动作维度、序列长度、观测维度、
        状态维度、奖励维度、目标维度、图像尺寸等
        （如 getActDim(), getObsDim(), getImgWidth()）。
        """
        self.act_dim = self.wrapper.getActDim()
        self.seq_dim = self.wrapper.getSeqDim()
        self.obs_dim = self.wrapper.getObsDim()#没怎么被利用
        self.state_dim = self.wrapper.getStateDim()
        self.rew_dim = self.wrapper.getRewDim()
        self.goal_obs_dim = self.wrapper.getGoalObsDim()#关系到状态这部分的观测
        self.img_width = self.wrapper.getImgWidth()
        self.img_height = self.wrapper.getImgHeight()
        #2) 观测/动作空间定义
        self._observation_space = spaces.Dict(
            {
                'image': spaces.Box(
                    low=0,
                    high=255,
                    shape=(n_seq, 256, 256),
                    dtype='uint8',
                ),
                'state': spaces.Box(
                    np.ones([n_seq, self.goal_obs_dim]) * -np.Inf,
                    np.ones([n_seq, self.goal_obs_dim]) * np.Inf,
                    dtype=np.float64,
                )
            }
        )

        self._action_space = spaces.Box(
            low=np.ones(self.act_dim) * -1.0,
            high=np.ones(self.act_dim) * 1.0,
            dtype=np.float64,
        )
        #3) 初始化观测缓存
        self._observation = {'image': np.zeros([self.num_envs, n_seq, self.img_height, self.img_width], dtype=np.uint8),#存每个环境的历史深度图序列
                            'state': np.zeros([self.num_envs, n_seq, self.goal_obs_dim], dtype=np.float64)}#存每个环境的状态序列
        self._state_observation = np.zeros([self.num_envs, self.goal_obs_dim], dtype=np.float64)#当前时刻的状态向量（没有序列维），每次 step() 从底层环境更新。
        self._observation_test = np.zeros([self.num_envs, self.obs_dim], dtype=np.float64)
        self._current_state = np.zeros([self.num_envs, self.state_dim], dtype=np.float64)#保存底层环境的完整状态（比 goal_obs_dim 更全的状态维度）
        self._rgb_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height * 3], dtype=np.uint8
        )
        self._gray_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.uint8
        )
        self._depth_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.float32
        )
        self.label_images = np.zeros([28, 28], dtype=np.float32)
        #4) 奖励与 done 缓存
        self._reward_components = np.zeros(
            [self.num_envs, n_seq, self.rew_dim], dtype=np.float64#存奖励分量的序列，形状是(环境数, 序列长度, 奖励维度)。
        )
        self._done = np.zeros((self.num_envs, n_seq), dtype=np.bool)#存终止标记的序列，形状 (环境数, 序列长度)，对应历史每一帧是否 done。
        self._single_reward_components = np.zeros(
            [self.num_envs, self.rew_dim], dtype=np.float64#存当前一步的奖励分量（没有序列维），由底层环境 step() 直接写入。
        )
        self._single_done = np.zeros((self.num_envs), dtype=np.bool)#存当前一步的 done 标记（每个环境一个）。
        #5) 额外信息和统计
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self.reward_names = self.wrapper.getRewardNames()
        self._extraInfo = np.zeros(
            [self.num_envs, len(self._extraInfoNames)], dtype=np.float64
        )
        self._extraInfo_test = np.zeros(
            [self.num_envs, len(self._extraInfoNames)], dtype=np.float64
        )

        self.rewards = [[] for _ in range(self.num_envs)]
        self.sum_reward_components = np.zeros(
            [self.num_envs, self.rew_dim - 1], dtype=np.float64
        )

        self._quadstate = np.zeros([self.num_envs, 14], dtype=np.float64)
        self._quadact = np.zeros([self.num_envs, self.act_dim], dtype=np.float64)
        self._flightmodes = np.zeros([self.num_envs, 1], dtype=np.float64)

        #  state normalization
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.obs_dim])
        self.obs_rms_new = RunningMeanStd(shape=[self.num_envs, self.obs_dim])
        self.max_episode_steps = 1000

        #6) 序列内存，用于维护 n_seq 历史帧，step()/reset() 里会滚动更新。
        self.image_memory = [[] for _ in range(self.num_envs)]
        self.state_memory = [[] for _ in range(self.num_envs)]
        self.reward_memory = [[] for _ in range(self.num_envs)]
        self.done_memory = [[] for _ in range(self.num_envs)]
        self.if_eval = False
        # VecEnv.__init__(self, self.num_envs,
        #                 self._observation_space, self._action_space)

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def update_rms(self):
        self.obs_rms = self.obs_rms_new

    def getLabelImg(self, depth):
        depth = (np.minimum(depth, 12.0)) / 12.0
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((28, 28))
        label = np.array(depth_img)
        return label

    def getLabelImage(self):
        return self.label_images

    """
    RecurrentPPO.collect_rollouts() 或类似方法会调用策略的 forward_rnn() 
    获取 latent_pi 和 latent_vf，然后调用 forward() 生成 actions。
    生成的 actions 被传递给环境的 step(action) 方法，作为输入执行环境交互。
    本质是把动作送进 AvoidBench 的 C++ 环境，让它原地写出 obs、reward、done、extra_info，
    而奖励的具体公式就是你贴出的 C++ AvoidVisionEnv::step 里那段计算。

    功能:
    接收动作，更新底层环境，渲染新深度图，滚动更新序列（删除最旧帧，追加新帧），返回更新后的 obs。
    执行一步环境交互：把动作送入底层环境，获取新的状态/深度图，更新序列观测，并返回 (obs, reward, done, info)。
    
    输入:
    action: 动作数组（若维度不足，会 reshape 成 (num_envs, act_dim)）。
    
    输出:
    obs: 字典，包含 image/state 的序列数据 (num_envs, n_seq, ...)。
    rewards: 每个环境的即时奖励（取 _single_reward_components 最后一项）。
    dones: 每个环境的终止标志。
    info: 额外信息（含 episode 回报统计）。
    
    调用关系:
    被 PPO 的 collect_rollouts() / collect_lstm_rollouts() 中每一步调用，用于生成训练数据。
    
    先调用 env.reset() 获取初始观测（obs），初始化环境状态和序列内存（例如填充 n_seq 帧的历史数据）。
    
    在 RecurrentPPO.train() 或 collect_rollouts() 中，先 obs = env.reset()，然后在循环中
    obs, reward, done, info = env.step(actions)。如果 done，则 obs = env.reset()。

    """
    def step(self, action):
        #1、动作整形
        if action.ndim <= 1:
            action = action.reshape((-1, self.act_dim))
        """
        更新当前时刻的状态/奖励/终止标志
        更新 _state_observation（当前状态向量）
        写入 _single_reward_components（奖励分量）
        写入 _single_done（终止标记）
        写入 _extraInfo（额外信息）
        """
        #2、调用底层环境执行一步
        self.wrapper.step(#
            action,
            self._state_observation,
            self._single_reward_components,
            self._single_done,
            self._extraInfo,
        )
        # update the mean and variance of the Running Mean STD
        # self.obs_rms_new.update(self._observation)
        t0 = time.time()
        self.render(0)
        # print("render time: ", time.time() - t0)
        #3、渲染并获取深度图
        depth = self.getDepthImage()#获取每个环境的深度图（扁平数组）。
        #4、更新序列观测（每个环境），删除序列最旧一帧，追加当前图像和当前状态
        for i in range(self.num_envs):
            img = depth[i, :].reshape(self.img_height, self.img_width)#把 depth[i] reshape 成 (H, W)。
            # depth_img = Image.fromarray((np.minimum(img, 12.0)) / 12.0 * 255.0)
            # if i==0:
            #     depth_img.convert('RGB').save('step'+str(i)+str(time.time())+'.jpg')
            img = self.preprocess(img)#preprocess()：限幅到 12m，再缩放到 0–255。
            del self.image_memory[i][:1]#滚动更新序列：
            del self.state_memory[i][:1]#从序列里删除最旧的一帧（实现滑动窗口）。

            self.image_memory[i].append(img.copy())
            self.state_memory[i].append(self._state_observation[i, :].copy())#把当前帧图像和当前状态追加到序列末尾。
        #5、组装新的 obs 字典
        self._observation['image'] = np.stack(self.image_memory)
        self._observation['state'] = np.stack(self.state_memory)
        obs = self._observation
        #6、构建 info 与 episode 统计
        info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._single_reward_components[i, -1])
            for j in range(self.rew_dim - 1):
                self.sum_reward_components[i, j] += self._single_reward_components[i, j]
            if self._single_done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                for j in range(self.rew_dim - 1):
                    epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                    self.sum_reward_components[i, j] = 0.0
                info[i]["episode"] = epinfo
                self.rewards[i].clear()
        #7、返回
        return (
            obs,
            self._single_reward_components[:, -1].copy(),# 取 _single_reward_components 的最后一项（即时奖励）
            self._single_done.copy(),
            info.copy(),
        )

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float64)
    
    def preprocess(self, image):
        depth = (np.minimum(image, 12.0)) / 12.0 * 255.0
        return depth.astype('int')

    """
    功能:
    重置环境，填充 image_memory 和 state_memory，并构造初始 obs 字典（含 image 和 state 的序列），用于训练/推理起始状态。
    
    输入:
    random: bool：是否随机重置底层环境。
    
    输出:
    obs: 字典，obs['image'] 和 obs['state'] 都是形状 (num_envs, n_seq, ...) 的序列数据。

    调用关系：
    在训练开始时由 RecurrentPPO 的上层逻辑间接调用（env.reset()）。
    在测试/评估时也会被调用以获取初始观测。

    训练/评估开始：先调用 env.reset() 获取初始观测（obs），初始化环境状态和序列内存（例如填充 n_seq 帧的历史数据）。
    交互循环：然后循环调用 env.step(action) 执行动作、获取新观测、奖励和终止标志，直到 episode 结束（done=True）。
    Episode 重置：当 done=True 时，再次调用 reset() 开始新 episode。
    """
    def reset(self, random=True):
        self.wrapper.reset(self._state_observation, random)
        # print(self._state_observation)
        self.render(0)
        self.render(0)
        depth = self.getDepthImage()
        for i in range(self.num_envs):
            img = depth[i, :].reshape(self.img_height, self.img_width)
            # depth_img = Image.fromarray((np.minimum(img, 12.0)) / 12.0 * 255.0)
            # if i==0:
            #     depth_img.convert('RGB').save('reset'+str(i)+str(time.time())+'.jpg')
            img = self.preprocess(img)
            del self.image_memory[i][:]
            del self.state_memory[i][:]
            self.image_memory[i] = [img.copy() for _ in range(n_seq)]
            self.state_memory[i] = [self._state_observation[i, :].copy() for _ in range(n_seq)]

        self._observation['image'] = np.stack(self.image_memory)
        self._observation['state'] = np.stack(self.state_memory)
        obs = self._observation
        return obs

    def resetRewCoeff(self):
        return self.wrapper.resetRewCoeff()

    def getObs(self):
        return self._observation

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def get_obs_norm(self):
        return self.obs_rms.mean, self.obs_rms.var

    def getProgress(self):
        return self._reward_components[:, 0]

    """
    如果需要深度图（用于观测），应调用 getDepthImage() 而非 getImage。
    在项目中，观测主要基于深度图（obs['image']），RGB/灰度图像更多用于辅助。
    """
    def getImage(self, rgb=False):
        if rgb:
            self.wrapper.getImage(self._rgb_img_obs, True)
            return self._rgb_img_obs.copy()
        else:
            self.wrapper.getImage(self._gray_img_obs, False)
            return self._gray_img_obs.copy()

    """
    从底层环境（self.wrapper，
    如 Unity 或 Gazebo 仿真器）获取深度图像数据，用于提供环境的视觉观测。
    """
    def getDepthImage(self):
        has_img = False
        # while(not has_img):
        has_img = self.wrapper.getDepthImage(self._depth_img_obs)
            # time.sleep(0.01)
        return self._depth_img_obs.copy()

    def getPointClouds(self, dir, id, save_pc):
        self.wrapper.getPointClouds(dir, id, save_pc)

    def readPointClouds(self, id):
        self.wrapper.readPointClouds(id)

    def getSavingState(self):
        return self.wrapper.getSavingState()

    def getReadingState(self):
        return self.wrapper.getReadingState()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(
            action,
            self._observation,
            self._reward,
            self._done,
            self._extraInfo,
            send_id,
        )

        return receive_id

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to unnormalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: unnormalized observation
        """
        return (obs * np.sqrt(obs_rms.var + 1e-8)) + obs_rms.mean

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        # obs_ = deepcopy(obs)
        # print(self.obs_rms.var)
        obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float64)
        return obs_

    def getQuadState(self):
        self.wrapper.getQuadState(self._quadstate)
        return self._quadstate

    def getQuadAct(self):
        self.wrapper.getQuadAct(self._quadact)
        return self._quadact

    def getExtraInfo(self):
        return self._extraInfo

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            for j in range(self.rew_dim - 1):
                epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                self.sum_reward_components[i, j] = 0.0
            info[i]["episode"] = epinfo
            self.rewards[i].clear()
        return info

    def close(self):
        self.wrapper.close()
        
    def render(self, frame_id, mode="human"):
        return self.wrapper.updateUnity(frame_id)

    def connectUnity(self):
        return self.wrapper.connectUnity()
    
    def initializeConnections(self):
        self.wrapper.initializeConnections()

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    def spawnObstacles(self, change_obs, seed=-1, radius=-1.0):
        self.wrapper.spawnObstacles(change_obs, seed, radius)
    
    def ifSceneChanged(self):
        return self.wrapper.ifSceneChanged()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError("This method is not implemented")

    def stop_recording_video(self):
        raise RuntimeError("This method is not implemented")

    def step_async(self):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError("This method is not implemented")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError("This method is not implemented")

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "VecNormalize":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def save_rms(self, save_dir, n_iter) -> None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = save_dir + "/iter_{0:05d}".format(n_iter)
        np.savez(
            data_path,
            mean=np.asarray(self.obs_rms.mean),
            var=np.asarray(self.obs_rms.var),
        )

    def load_rms(self, data_dir) -> None:
        self.mean, self.var = None, None
        np_file = np.load(data_dir)
        #
        self.mean = np_file["mean"]
        self.var = np_file["var"]
        #
        self.obs_rms.mean = np.mean(self.mean, axis=0)
        self.obs_rms.var = np.mean(self.var, axis=0)