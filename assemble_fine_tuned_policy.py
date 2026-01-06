import os
import torch
from stable_baselines3.common.utils import get_device
from rpg_baselines_prev.torch.recurrent_ppo.policies import MultiInputLstmPolicy

device = get_device("auto")
weight = os.environ["AVOIDER_PATH"] + "/../learning/saved/RecurrentPPO_{0}/Policy/iter_{1:05d}.pth".format(250, 700)
weight_fine_tune_part = os.environ["AVOIDER_PATH"] + "/../learning/saved/RecurrentPPO_1/Policy/iter_{0:05d}.pth".format(20)
saved_varables = torch.load(weight, map_location=device)
saved_varables_fine_tune_part = torch.load(weight_fine_tune_part, map_location=device)
for name in saved_varables["state_dict"].keys():
    print(name)
policy = MultiInputLstmPolicy(features_dim=64, **saved_varables["data"])
policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
saved_varables["state_dict"]["features_extractor.conv1.weight"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv1.weight"]
saved_varables["state_dict"]["features_extractor.conv1.bias"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv1.bias"]
saved_varables["state_dict"]["features_extractor.conv2.weight"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv2.weight"]
saved_varables["state_dict"]["features_extractor.conv2.bias"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv2.bias"]
saved_varables["state_dict"]["features_extractor.conv3.weight"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv3.weight"]
saved_varables["state_dict"]["features_extractor.conv3.bias"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv3.bias"]
saved_varables["state_dict"]["features_extractor.conv4.weight"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv4.weight"]
saved_varables["state_dict"]["features_extractor.conv4.bias"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv4.bias"]
saved_varables["state_dict"]["features_extractor.conv5.weight"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv5.weight"]
saved_varables["state_dict"]["features_extractor.conv5.bias"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv5.bias"]
saved_varables["state_dict"]["features_extractor.conv6.weight"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv6.weight"]
saved_varables["state_dict"]["features_extractor.conv6.bias"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.conv6.bias"]
saved_varables["state_dict"]["features_extractor.linear.weight"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.linear.weight"]
saved_varables["state_dict"]["features_extractor.linear.bias"] = saved_varables_fine_tune_part["state_dict"]["features_extractor.linear.bias"]
saved_varables["state_dict"]["lstm_actor.weight_ih_l0"] = saved_varables_fine_tune_part["state_dict"]["lstm_actor.weight_ih_l0"]
saved_varables["state_dict"]["lstm_actor.weight_hh_l0"] = saved_varables_fine_tune_part["state_dict"]["lstm_actor.weight_hh_l0"]
saved_varables["state_dict"]["lstm_actor.bias_ih_l0"] = saved_varables_fine_tune_part["state_dict"]["lstm_actor.bias_ih_l0"]
saved_varables["state_dict"]["lstm_actor.bias_hh_l0"] = saved_varables_fine_tune_part["state_dict"]["lstm_actor.bias_hh_l0"]
saved_varables["state_dict"]["mu_linear.weight"] = saved_varables_fine_tune_part["state_dict"]["mu_linear.weight"]
saved_varables["state_dict"]["mu_linear.bias"] = saved_varables_fine_tune_part["state_dict"]["mu_linear.bias"]
saved_varables["state_dict"]["feature_decoder0.fc.weight"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.fc.weight"]
saved_varables["state_dict"]["feature_decoder0.fc.bias"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.fc.bias"]
saved_varables["state_dict"]["feature_decoder0.deconv1.weight"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv1.weight"]
saved_varables["state_dict"]["feature_decoder0.deconv1.bias"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv1.bias"]
saved_varables["state_dict"]["feature_decoder0.deconv2.weight"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv2.weight"]
saved_varables["state_dict"]["feature_decoder0.deconv2.bias"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv2.bias"]
saved_varables["state_dict"]["feature_decoder0.deconv3.weight"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv3.weight"]
saved_varables["state_dict"]["feature_decoder0.deconv3.bias"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv3.bias"]
saved_varables["state_dict"]["feature_decoder0.deconv4.weight"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv4.weight"]
saved_varables["state_dict"]["feature_decoder0.deconv4.bias"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv4.bias"]
saved_varables["state_dict"]["feature_decoder0.deconv5.weight"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv5.weight"]
saved_varables["state_dict"]["feature_decoder0.deconv5.bias"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv5.bias"]
saved_varables["state_dict"]["feature_decoder0.deconv6.weight"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv6.weight"]
saved_varables["state_dict"]["feature_decoder0.deconv6.bias"] = saved_varables_fine_tune_part["state_dict"]["feature_decoder0.deconv6.bias"]
# saved_varables["state_dict"].pop("log_std")
policy.load_state_dict(saved_varables["state_dict"], strict=False)
policy.to(device)
policy.save(os.environ["AVOIDER_PATH"] + "/../learning/saved/best.pth")