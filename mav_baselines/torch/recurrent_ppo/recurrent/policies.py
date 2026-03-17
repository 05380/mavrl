from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import warnings
from gym import spaces
from functools import partial
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from mav_baselines.torch.recurrent_ppo.recurrent.type_aliases import RNNStates
from mav_baselines.torch.recurrent_ppo.recurrent.rnn_extractor import MultiExtractor, Encoder, Decoder
from mav_baselines.torch.recurrent_ppo.recurrent.beta_distribution import BetaDistribution, make_proba_distribution

class RecurrentActorCriticPolicy(ActorCriticPolicy):#在基础A-C基础上增加了lstm
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,#新加段
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        #新加段

        lstm_hidden_size: int = 256,# LSTM隐藏层单元数（输出维度）
        n_lstm_layers: int = 1,# LSTM层数
        shared_lstm: bool = False,# 是否共享LSTM（策略和价值网络共用）
        enable_critic_lstm: bool = True,# 是否为价值网络启用单独的LSTM
        states_dim: int = 6, # 状态向量维度
        features_dim: int = 32,
        only_lstm_training: bool = False,# 仅LSTM训练模式
        use_beta: bool = False, # 是否使用Beta分布
        reconstruction_members: Optional[List[bool]] = None,# 重建组件开关
        reconstruction_steps: int = 2,# 重建步数（过去/现在/未来帧）
        lstm_kwargs: Optional[Dict[str, Any]] = None,# LSTM其他参数
    ):
        #记录 LSTM 输出维度，是否用 Beta 动作分布。
        self.lstm_output_dim = lstm_hidden_size
        self.use_beta = use_beta
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.states_dim = states_dim
        self.features_dim = features_dim
        self.only_lstm_training = only_lstm_training
        self.share_features_extractor = share_features_extractor
        self.reconstruction_members = reconstruction_members
        self.reconstruction_steps = reconstruction_steps
        # if self.share_features_extractor:
        #     self.vf_features_extractor = self.features_extractor
        # else:
        #     self.vf_features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        #LSTM 配置
        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        ## LSTM 网络构建，处理图像信息的神经网络 - 作为后续forward_rnn（）中的_process_sequence（）的lstm网络架构参数 将图像特征加上lstm
        self.lstm_actor = nn.LSTM(
            self.features_dim + states_dim,#64 + 0,输入: CNN特征
            lstm_hidden_size,
            num_layers=n_lstm_layers,# 输出: 256维表示
            **self.lstm_kwargs,
        )
        # 用于预测未来帧的线性层
        self.mu_linear = nn.Linear(lstm_hidden_size, 3 * (self.features_dim + states_dim))
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        # LSTM隐藏状态形状定义
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)
        # # 价值LSTM（可选）- 用于价值评估
        #这个项目使用共享 LSTM 策略：
        # shared_lstm = True
        # → 只用 Actor LSTM，Critic 使用 Actor 输出 + detach()
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                self.features_dim + states_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
        # 解码器 - 用于特征重建，用于重建图像特征（past/now/future），服务于 LSTM 预测任务。
        self.feature_decoder0 = Decoder(self.observation_space, self.features_dim + states_dim)
        # self.feature_decoder1 = Decoder(self.observation_space, self.features_dim + states_dim)

        # 用 schedule 的初始学习率创建优化器。
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
   
    """
    是策略网络的搭建阶段，负责把“MLP 主干 + 动作分布头 + 价值头 + 优化器”全部创建好。
    param lr_schedule: 学习率调度器，用于设置初始学习率  
   
    调用 _build_mlp_extractor() 创建 actor/critic 的 MLP 主干。
    根据动作分布类型（高斯/离散/Beta/…）构造对应的 action head。
    创建 value_net（线性层）输出价值。
    可选进行正交初始化。
    创建优化器。

    在 RecurrentActorCriticPolicy 初始化过程中被调用：
    __init__() → super().__init__(...) →（父类 ActorCriticPolicy）_build()。
    后续 forward() / evaluate_actions() 使用这里创建的 action/value head。
    
    """
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        if self.use_beta:
            self.action_dist = make_proba_distribution(self.action_space)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
            # print("self.log_std: ", self.log_std)
            # self.log_std = th.tensor(0.0, dtype=th.float32, device=self.device)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BetaDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
    """
    创建策略/价值的 MLP 主干，并把它挂在 self.mlp_extractor 上，
    供后续动作头/价值头使用。

    被 _build() 调用（见同文件 _build()），
    而 _build() 在策略初始化时由父类 ActorCriticPolicy 触发。

    训练/推理时，forward()、evaluate_actions() 
    会调用 self.mlp_extractor.forward_actor/forward_critic，依赖这里创建的 MLP。

    """
    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim + 8,#输入维度原来是+7
            net_arch=self.net_arch,#来自 policy_kwargs，决定 Actor/ Critic MLP 的层数与宽度。
            activation_fn=self.activation_fn,#激活函数，比如 ReLU/Tanh。
            device=self.device,
        )

    """
    功能:
    把 batch 维度重排为序列维度，处理 LSTM 的时序输入；
    若 episode_starts 中间有重置点，就逐步循环并对 hidden state 做清零处理；
    最后把序列输出再展平成 batch。

    输入:
    features: LSTM 的输入特征，shape 通常是 (batch, lstm.input_size)
    lstm_states: 上一时刻的 (h, c)
    episode_starts: 是否新 episode，用于在序列中重置状态
    lstm: 要用的 LSTM 模块（actor 或 critic）

    输出:
    lstm_output: 展平后的 LSTM 输出，shape (batch, lstm_hidden_size)
    lstm_states: 更新后的 (h, c)

    被 forward_rnn() 调用（用于 actor LSTM / critic LSTM）。

    """
    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.LSTM,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.
        :param features: Input tensor
        :param lstm_states: previous cell and hidden states of the LSTM
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1] # 从隐状态获取序列数
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        ## reshape: 将batch维度重新解释为 (n_seq, max_length)；# 交换轴：[n_seq, max_length, feature_dim] → [max_length, n_seq, feature_dim]
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states
    
    """
    功能:
    image → extract_features() 得到图像特征
    state reshape 成状态向量
    图像特征送入 LSTM（_process_sequence()）得到 latent_pi / latent_vf
    将 LSTM 输出与状态向量拼接，得到最终的融合特征

    输入:
    obs: 观测字典，包含 image 与 state
    lstm_states: actor/critic 的 LSTM 状态
    episode_starts: episode 起始标记

    输出:
    latent_pi: 用于策略分支的融合特征
    latent_vf: 用于价值分支的融合特征
    RNNStates(...): 更新后的 LSTM states（actor/critic）

    调用关系:
    在 RecurrentPPO.collect_rollouts() / collect_lstm_rollouts() 中被调用，用于生成动作与价值所需的特征。
    返回的 latent_pi/latent_vf 会传给 forward() 进一步得到动作分布与价值。
    """
    def forward_rnn(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        cat_pi = []
        cat_vf = []
        for key, _obs in obs.items():
            if key == 'image':#图像数据处理
                """
                extract_features() 用的是 policy 自己的特征提取器
                （例如 rnn_extractor.Encoder）。
                只有在 state_vae 传入时，会把 VAE encoder 的权重拷贝
                进这个 Encoder，但此时 PPO 不会调用 VAE 模型本身。
                """
                features = self.extract_features(_obs)
                if self.share_features_extractor:#11共享模式，策略和价值网络使用相同特征表示
                    pi_features = vf_features = features
                else:#独立模式：策略网络: pi_features 用于动作选择；价值网络: vf_features 用于状态价值估计
                    pi_features, vf_features = features
            else:#状态向量
                #状态重塑：原始: [4, n_seq, 7] 4个环境, 多步序列, 7维状态；重塑: [4, 7] 最后一个时间步的状态
                state_shape = _obs.shape
                # 包含: [log_distance, 水平速度, 目标方向, 速度方向, 
                #高度差, 竖直速度, 偏航角]
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()#这里得到了状态特征
                #pi_features = th.cat([pi_features, _obs[:, 4:]], dim=1)
                # vf_features = th.cat([vf_features, _obs[:, 4:]], dim=1)


                #针对策略网络Actor 进行LSTM序列处理，返回更新后的LSTM隐藏状态
                #latent_pi：Actor LSTM网络处理后的隐状态输出，LSTM对融合特征的时序处理
                # lstm_states_pi：LSTM处理完当前时间步后的新隐状态和细胞状态保存用于下一步
                latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi,
                                            #让 LSTM 专注于图像序列的时序关系，因为在 train_policy.py 里构建模型时传了 states_dim=0      
                                            episode_starts, self.lstm_actor)#这里的pi_features传入的是图像特征，而非融合特征，
                """
                针对价值网络
                三种价值网络处理方式：
                独立LSTM: 使用单独的价值LSTM网络
                共享LSTM: 使用策略LSTM输出，但切断梯度
                线性层: 使用简单的线性变换
                """
                if self.lstm_critic is not None:
                    latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf,
                                            episode_starts, self.lstm_critic)
                elif self.shared_lstm:#实际使用
                    """ 
                    Critic 共用 Actor 的 LSTM 输出 — 为了节省计算，价值网络直接使用策略网络的 LSTM 隐藏状态
                    但梯度只从 Actor 反向传播 — detach() 防止价值网络的梯度流向 LSTM，确保：
                        LSTM 只根据 Actor 的损失函数更新（通常是 PPO 策略损失）
                        Critic 的损失不会影响 LSTM 权重
                    解耦两个网络的优化 — 虽然使用相同的特征，但各自独立优化"""
                    latent_vf = latent_pi.detach()
                    lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
                else:
                    latent_vf = self.critic(vf_features)
                    lstm_states_vf = lstm_states_pi
                """拼接处理后的特征和状态向量，形成最终的输入特征"""
                cat_pi = [latent_pi, _obs] # 策略特征 = [LSTM输出, 状态向量]
                cat_vf = [latent_vf, _obs]# 价值特征 = [LSTM输出, 状态向量]
        latent_pi = th.cat(cat_pi, dim=1) # 拼接维度1：[batch, 256+7]
        latent_vf = th.cat(cat_vf, dim=1) # 拼接维度1：[batch, 256+7]
        return latent_pi, latent_vf, RNNStates(lstm_states_pi, lstm_states_vf)
    
    """
    功能:
    mlp_extractor 将 latent_pi/latent_vf 映射为高层特征；
    value_net 输出价值；
    根据 latent_pi_ 构建动作分布并采样动作；
    计算该动作的 log_prob。

    输入:
    latent_pi: 策略分支的融合特征（来自 forward_rnn()）
    latent_vf: 价值分支的融合特征（来自 forward_rnn()）
    deterministic: 是否用确定性动作（评估时常用）

    输出:
    actions: 采样/确定性的动作
    values: 价值估计 V(s)
    log_prob: 动作对数概率（PPO 计算损失用）

    调用关系:
    在 RecurrentPPO.collect_rollouts() 中被调用：
    forward_rnn() 先得到 latent_pi/latent_vf
    再调用 forward() 得到 actions/values/log_prob
    训练时用于生成 rollout 数据和 PPO 损失所需量。

    PPO 在线决策时只做 编码（Encoder）→ LSTM → MLP → 动作分布。
    解码（Decoder）只在 LSTM 重建训练路径（如 train_lstm_from_dataset() /
    predict_lstm()）里用来重建图像，不参与动作生成。

    原始图像
    → extract_features()（Encoder）
    → LSTM（forward_rnn）
    → MLP（forward）
    → _get_action_dist_from_latent()
    → 采样动作
    """
    def forward(self, latent_pi: th.Tensor, latent_vf: th.Tensor,deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        前置步骤（forward_rnn 方法）：
        观测（图像 + 状态）经过 CNN 提取图像特征。
        图像特征送入 LSTM 处理时序关系，得到 latent_pi（LSTM 输出 + 状态拼接）。
        latent_pi 传递给 forward 方法。

        ：原始 latent_pi 可能包含噪声或冗余信息，
        MLP 通过多层变换提取关键模式，提高决策质量。
        #将LSTM输出变换为适合生成动作分布的特征向量，给策略分支用的高层特征
        在 Actor-Critic 架构中，策略网络（Actor）需要将低级特征
        （如 CNN 提取的图像特征 + LSTM 处理的时序特征 + 状态向量）
        转换为适合动作分布（例如高斯分布或 Beta 分布）的输入。

        内部过程：
        应用线性层 + 激活函数（默认 nn.Tanh）的多层变换。

        forward_actor 方法执行多层感知机（MLP）的正向传播，
        应用非线性变换（如激活函数），以提取更抽象的特征，提高策略的表达能力。
        这一步还是属于actor网络中干的事情
        """
        latent_pi_ = self.mlp_extractor.forward_actor(latent_pi)
        #为价值函数生成输入特征，给价值分支用的高层特征。
        latent_vf_ = self.mlp_extractor.forward_critic(latent_vf)
        # 计算状态价值函数
        values = self.value_net(latent_vf_)
        #获取动作分布
        distribution = self._get_action_dist_from_latent(latent_pi_)
        #采样动作
        actions = distribution.get_actions(deterministic=deterministic)
        #PPO算法中的策略梯度计算，输出：该动作的对数概率 [batch_size]
        log_prob = distribution.log_prob(actions)
        # print("log_prob: ", log_prob)
        return actions, values, log_prob

    def forward_rnn_cmaes(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        cat_pi = []
        for key, _obs in obs.items():
            if key == 'image':
                features = self.extract_features(_obs)
            else:
                state_shape = _obs.shape
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()

                latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states,
                                            episode_starts, self.lstm_actor)

                cat_pi = [latent_pi, _obs]
        latent_pi = th.cat(cat_pi, dim=1)
        return latent_pi, lstm_states_pi
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        elif isinstance(self.action_dist, BetaDistribution):
            return self.action_dist.proba_distribution(action_logits=(mean_actions+1.0))
        else:
            raise ValueError("Invalid action distribution")

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        # latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)

        cat_pi = []
        for key, _obs in obs.items():
            if key == 'image':
                features = self.extract_features(_obs)
            else:
                state_shape = _obs.shape
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()
                # features = th.cat([features, _obs[:, 4:]], dim=1)
                latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
                cat_pi = [latent_pi, _obs]
        latent_pi = th.cat(cat_pi, dim=1)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states


    """
    功能:
    对图像观测做必要的预处理（如归一化），然后用特征提取器（CNN Encoder）提取特征向量。

    输入:
    obs: th.Tensor：图像观测张量（来自 obs['image']）。
    features_extractor: Optional[BaseFeaturesExtractor]：可选自定义特征提取器；为空则用 self.features_extractor。

    输出:
    th.Tensor：图像特征向量（维度为 features_dim）。

    调用关系:
    在 forward_rnn() / get_distribution() / predict_values() 中处理 obs['image'] 时调用。
    to_latent() 也会调用它来获取图像的 latent 表示。

    trainvae.py 训练出 VAE（含 encoder 权重）。
    训练 PPO 时，如果 state_vae 被传进来：
    把 VAE encoder 权重 加载到 policy 的 Encoder（作为初始化）。
    然后 PPO 训练继续更新这个 Encoder（除非你手动冻结）。
    """
    def extract_features(self, obs: th.Tensor, features_extractor: Optional[BaseFeaturesExtractor] = None) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.
         :param obs: The observation
         :param features_extractor: The features extractor to use. If it is set to None,
            the features extractor of the policy is used.
         :return: The features
        """
        if features_extractor is None:
            warnings.warn(
                (
                    "When calling extract_features(), you should explicitely pass a features_extractor as parameter. "
                    "This will be mandatory in Stable-Baselines v1.8.0"
                ),
                DeprecationWarning,
            )
        #特征提取器选择，如果没有提供，则使用实例的默认特征提取器
        features_extractor = features_extractor or self.features_extractor
        assert features_extractor is not None, "No features extractor was set"
        #从观测空间字典中获取图像部分的空间定义
        observation_space = self.observation_space['image']
        #观测预处理，根据 normalize_images 参数决定是否归一化图像
        preprocessed_obs = preprocess_obs(obs, observation_space, normalize_images=self.normalize_images)
        #self.features_extractor 的具体实现类，在 RecurrentMultiInputActorCriticPolicy 里作为默认的 features_extractor_class
        return features_extractor(preprocessed_obs)

    """
    功能:
    当模型已训练好，推理时我们使用 predict() 方法
    来生成动作。
    根据当前观测估计状态价值 (V(s))。

    输入:
    obs: th.Tensor：观测字典（image + state）。
    lstm_states: LSTM 的隐藏/细胞状态（用于 critic）。
    episode_starts: episode 起始标记，用于必要时重置 LSTM 状态。

    输出:
    th.Tensor：价值估计，shape 通常是 (batch, 1)。

    主要流程:
    image → extract_features() 得到图像特征。
    state → reshape 成状态向量。
    用 LSTM（或线性层）处理图像特征得到 latent_vf。
    将 latent_vf 与状态向量拼接。
    mlp_extractor.forward_critic() → value_net 得到价值。

    调用关系:
    在 collect_rollouts() 的末尾用于 bootstrap 最后一步的 value。
    调用mlp_extractor.forward_critic() 和 value_net 输出价值。
    """
    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        cat_vf = []
        for key, _obs in obs.items():
            if key == 'image':
                features = self.extract_features(_obs, self.features_extractor)
            else:
                state_shape = _obs.shape
                _obs = _obs.reshape([state_shape[0], state_shape[2]]).float()
                # features = th.cat([features, _obs[:, 4:]], dim=1)
                if self.lstm_critic is not None:
                    latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
                elif self.shared_lstm:
                    latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
                    latent_vf = latent_pi.detach()
                else:
                    latent_vf = self.critic(features)
                cat_vf = [latent_vf, _obs]
        latent_vf = th.cat(cat_vf, dim=1)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)
    
    """
    功能:
    用给定动作在当前策略下重新计算：价值、动作对数概率、熵（用于 PPO 损失）

    输入:
    latend_lstm_pi: 策略分支的融合特征（来自 LSTM/拼接后的特征）
    latend_lstm_vf: 价值分支的融合特征
    actions: 真实执行过的动作（rollout 里存的）

    输出:
    values: 价值估计 V(s)
    log_prob: 给定动作在当前策略下的对数概率
    entropy: 动作分布熵（用于探索正则）

    调用关系:
    在 RecurrentPPO.train() 里被调用，用于计算 PPO 的 policy loss、value loss、entropy bonus。
    """
    def evaluate_actions(self, latend_lstm_pi: th.Tensor, latend_lstm_vf: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi = self.mlp_extractor.forward_actor(latend_lstm_pi)
        latent_vf = self.mlp_extractor.forward_critic(latend_lstm_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        # print("actions: ", actions)
        # print("log_prob: ", log_prob)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    """
    功能:
    给定观测和 LSTM 状态，计算策略动作并返回更新后的 LSTM 状态。

    输入:
    observation: th.Tensor：观测（字典或张量，已转成 torch）。
    lstm_states: LSTM 的 (h, c) 状态。
    episode_starts: episode 起始标记，用于重置 LSTM。
    deterministic: 是否使用确定性动作。

    输出:
    actions: 采样/确定性的动作。
    lstm_states: 更新后的 LSTM 状态。

    调用关系:
    被 predict() 调用：predict() 负责把 numpy 转成 tensor、处理 batch 维后再调用 _predict()。
    """
    
    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states

    """
    功能:
    推理接口：给定观测（numpy），返回动作（numpy）和下一步 LSTM 状态；处理输入格式、状态初始化、设备转换、动作后处理等。

    输入:
    observation: np.ndarray 或 Dict[str, np.ndarray]，原始观测（图像/状态字典）。
    state: 可选 LSTM 状态 (h, c)；为空则自动初始化为零。
    episode_start: 可选 episode 起始标记；为空则默认全 False。
    deterministic: 是否输出确定性动作。

    输出:
    actions: np.ndarray 动作（已裁剪到动作空间范围）。
    states: 下一步 LSTM 状态 (h, c)，用于下一次调用。

    调用关系:
    调用 _predict() 得到动作和新 LSTM 状态。
    _predict() 内部调用 get_distribution() → forward_rnn() → _get_action_dist_from_latent()。
    """
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
                state[1], dtype=th.float32, device=self.device
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation, lstm_states=states, episode_starts=episode_starts, deterministic=deterministic
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()
        # print(actions)
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states
            
    #将观测转换到潜在空间， 主要用于LSTM训练和重建任务，
    #在 RecurrentPPO.train_lstm() / train_lstm_from_dataset() 等 LSTM 重构训练路径中调用，用于得到固定的图像 latent。
    def to_latent(self, obs):
        with th.no_grad():
            if isinstance(obs, dict):
                obs_mu = self.extract_features(obs['image'])
            else:
                obs_mu = self.extract_features(obs)
        # latent_obs, latent_next_obs = [
        #     (x_mu + x_logsigma.exp() * th.randn_like(x_mu))
        #     for x_mu, x_logsigma in [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        # latent_obs = th.cat([obs_mu, obs['state'].squeeze().float()[:, 3:]], dim=1)
        return obs_mu
            
    def predict_img(self,
        latent_obs: np.ndarray):
        # Switch to eval mode (this affects batch norm / dropout)
        with th.no_grad():
            latent_obs = th.tensor(latent_obs, dtype=th.float32, device=self.device)
            latent_obs = self.mu_linear(latent_obs)
            recon_latent_size = self.features_dim + self.states_dim
            pre_latent_obs, cur_latent_obs, next_latent_obs = th.split(latent_obs, [recon_latent_size, recon_latent_size, recon_latent_size], dim=1)
            total_laten_obs = [pre_latent_obs, cur_latent_obs, next_latent_obs]
            reconstruction = []
            for i in range(len(self.reconstruction_members)):
                if self.reconstruction_members[i]:
                    reconstruction.append(self.feature_decoder0(total_laten_obs[i]).cpu().numpy())
                else:
                    reconstruction.append(None)
        return reconstruction

    """
    
    """
    def predict_lstm(self, 
        latent_obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        is_eva: bool = False,
        ) -> Tuple[th.Tensor, th.Tensor, int, Tuple[th.Tensor, th.Tensor]]:

        pre_latent_obs, lstm_state = self._process_sequence(latent_obs, lstm_states, episode_starts, self.lstm_actor)
        n_seq = lstm_states[0].shape[1]
        pre_latent_obs = pre_latent_obs.reshape([n_seq, -1, self.lstm_output_dim])
        pre_latent_obs = self.mu_linear(pre_latent_obs)
        recon_latent_size = self.features_dim + self.states_dim
        pre_latent_obs, cur_latent_obs, next_latent_obs = th.split(pre_latent_obs, [recon_latent_size, recon_latent_size, recon_latent_size], dim=2)
        # reconstruction0 = self.feature_decoder0(th.flatten(cur_latent_obs, start_dim=0, end_dim=1))
        total_laten_obs = [pre_latent_obs, cur_latent_obs, next_latent_obs]
        reconstruction = []
        if is_eva:
            for i in range(len(self.reconstruction_members)):
                if self.reconstruction_members[0]:
                    reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[i], start_dim=0, end_dim=1)))
                else:
                    reconstruction.append(None)
                if self.reconstruction_members[1]:
                    reconstruction.append(self.feature_decoder1(th.flatten(total_laten_obs[i], start_dim=0, end_dim=1)))
                else:
                    reconstruction.append(None)
                if self.reconstruction_members[2]:
                    reconstruction.append(self.feature_decoder2(th.flatten(total_laten_obs[i], start_dim=0, end_dim=1)))
                else:
                    reconstruction.append(None)
        else:
            if self.reconstruction_members[0]:
                reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[0][:, self.reconstruction_steps:, :], start_dim=0, end_dim=1)))
            else:
                reconstruction.append(None)
            if self.reconstruction_members[1]:
                reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[1], start_dim=0, end_dim=1)))
            else:
                reconstruction.append(None)
            if self.reconstruction_members[2]:
                reconstruction.append(self.feature_decoder0(th.flatten(total_laten_obs[2][:, :-self.reconstruction_steps, :], start_dim=0, end_dim=1)))
            else:
                reconstruction.append(None)

        return reconstruction, n_seq, lstm_state


class RecurrentActorCriticCnnPolicy(RecurrentActorCriticPolicy):
    """
    CNN recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        states_dim: int = 0,
        features_dim: int = 32,
        only_lstm_training: bool = False,
        use_beta: bool = False,
        reconstruction_members: Optional[List[bool]] = None,
        reconstruction_steps: int = 2,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            states_dim,
            features_dim,
            only_lstm_training,
            use_beta,
            reconstruction_members,
            reconstruction_steps,
            lstm_kwargs,
        )


class RecurrentMultiInputActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(#传递所有参数给基类进行初始化
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        ##特征提取器（默认使用Encoder），适用场景: 同时处理图像和状态向量等混合输入，而父类默认: FlattenExtractor
        features_extractor_class: Type[BaseFeaturesExtractor] = Encoder, 
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,#LSTM隐藏层单元数量（默认256）
        n_lstm_layers: int = 1,#LSTM层数（默认1层）
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        states_dim: int = 0,
        features_dim: int = 32,
        only_lstm_training: bool = False,
        use_beta: bool = False,
        reconstruction_members: Optional[List[bool]] = None,
        reconstruction_steps: int = 2,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(#通过 super() 调用父类构造函数，
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            states_dim,
            features_dim,
            only_lstm_training,
            use_beta,
            reconstruction_members,
            reconstruction_steps,
            lstm_kwargs,
        )
