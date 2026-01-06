#!/usr/bin/env python3
import argparse
import os
from os.path import join, exists
import torch
from stable_baselines3.common.utils import get_device
from mav_baselines.torch.recurrent_ppo.policies import MultiInputLstmPolicy
from mav_baselines.torch.recurrent_ppo.ppo_recurrent import RecurrentPPO

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--retrain", type=int, default=1, help="if retrain")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    parser.add_argument("--dir", type=str, default="./saved/lstm_dataset",
                      help="Where to place rollouts")
    parser.add_argument("--recon", nargs='+', type=int, default=[0, 0, 1],
                      help="past now future")
    parser.add_argument("--lstm_exp", type=str, default="LSTM", help="lstm experiment name")
    return parser

def main():
    args = parser().parse_args()

    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    device = get_device("auto")
    weight = os.environ["AVOIDBENCH_PATH"] + "/../mavrl/saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
    device = get_device("auto")
    saved_variables = torch.load(weight, map_location=device)
    # print(saved_variables["state_dict"])
    # Create policy object
    saved_variables["data"]['only_lstm_training'] = True
    policy = MultiInputLstmPolicy(features_dim=64, 
                                  reconstruction_members=args.recon,
                                  reconstruction_steps=2,
                                  **saved_variables["data"])
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    policy.load_state_dict(saved_variables["state_dict"], strict=False)
    policy.to(device)
    print(args.recon)

    logdir = os.environ["AVOIDBENCH_PATH"] + "/../mavrl/exp_dir"
    vae_file = join(logdir, 'vae', 'best.tar')
    assert exists(vae_file), "No trained VAE in the logdir..."
    state_vae = torch.load(vae_file)
    print("Loading VAE at epoch {} "
        "with test error {}".format(state_vae['epoch'], state_vae['precision']))

    if args.train:
        model = RecurrentPPO(
            tensorboard_log=log_dir,
            policy=policy,
            policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[512, 512])],
            ),
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=1000,
            n_seq=1,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            lstm_layer=1,
            batch_size=500,
            n_epochs=2000,
            clip_range=0.2,
            use_sde=False,  # don't use (gSDE), doesn't work
            retrain=args.retrain,
            verbose=1,
            only_lstm_training=True,
            state_vae = state_vae,
            states_dim=0,
            reconstruction_members=args.recon,
            train_lstm_without_env=False,
            fine_tune_from_rosbag=True,
            lstm_dataset_path=args.dir,
            lstm_weight_saved_path=args.lstm_exp,
        )
        model.fine_tune_lstm_from_rosbag()

if __name__ == "__main__":
    main()