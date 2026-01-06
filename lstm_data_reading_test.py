import argparse
from os.path import join, exists
from os import mkdir
import numpy as np

import torch
import torch.utils.data
from torchvision import transforms
from data.loaders import RolloutLSTMSequenceDataset

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

dataset_train = RolloutLSTMSequenceDataset('./saved/RecurrentPPO_16', device, train=True)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=1, shuffle=False, num_workers=0)

dataset_train.load_next_buffer()

for batch_idx, data in enumerate(train_loader):
    print(data[1][0].shape)