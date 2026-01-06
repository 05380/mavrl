import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboardX import SummaryWriter
import numpy as np
from os import listdir, path
from os.path import join, isdir

sr_101_files = []
sr_011_files = []
sr_001_files = []
sr_010_files = []
sr_100_files = []
sr_110_files = []

root = "./matlab/sr1"
for sd in listdir(root):
    if sd == 'sr_101':
        sr_101_files = listdir(join(root, sd))
    elif sd == 'sr_011':
        sr_011_files = listdir(join(root, sd))
    elif sd == 'sr_001':
        sr_001_files = listdir(join(root, sd))
    elif sd == 'sr_010':
        sr_010_files = listdir(join(root, sd))
    elif sd == 'sr_100':
        sr_100_files = listdir(join(root, sd))
    elif sd == 'sr_110':
        sr_110_files = listdir(join(root, sd))

success_rate = np.zeros((3, 50, 3))
file_count = 0
if len(sr_101_files) != 0:
    for f in sr_101_files:
        count = 0
        for event in summary_iterator(join(root, 'sr_101', f)):
            for value in event.summary.value:
                if count >= 50:
                    continue
                if "test/success_rate_medium" in value.tag:
                    success_rate[file_count, count, 0] = event.step
                    success_rate[file_count, count, 1] = value.simple_value
                    count += 1
                elif "test/success_rate_hard" in value.tag:
                    success_rate[file_count, count, 2] = value.simple_value
        file_count += 1
np.savetxt("./matlab/success_rate_101.csv", success_rate.reshape(-1, success_rate.shape[-1]), delimiter=" ")


success_rate = np.zeros((3, 50, 3))
file_count = 0
if len(sr_010_files) != 0:
    for f in sr_010_files:
        count = 0
        for event in summary_iterator(join(root, 'sr_010', f)):
            for value in event.summary.value:
                if count >= 50:
                    continue
                if "test/success_rate_medium" in value.tag:
                    success_rate[file_count, count, 0] = event.step
                    success_rate[file_count, count, 1] = value.simple_value
                    count += 1
                elif "test/success_rate_hard" in value.tag:
                    success_rate[file_count, count, 2] = value.simple_value
        file_count += 1
np.savetxt("./matlab/success_rate_010.csv", success_rate.reshape(-1, success_rate.shape[-1]), delimiter=" ")

success_rate = np.zeros((3, 50, 3))
file_count = 0
if len(sr_110_files) != 0:
    for f in sr_110_files:
        count = 0
        for event in summary_iterator(join(root, 'sr_110', f)):
            for value in event.summary.value:
                if count >= 45:
                    continue
                if "test/success_rate_medium" in value.tag:
                    success_rate[file_count, count, 0] = event.step
                    success_rate[file_count, count, 1] = value.simple_value
                    count += 1
                elif "test/success_rate_hard" in value.tag:
                    success_rate[file_count, count, 2] = value.simple_value
        file_count += 1
np.savetxt("./matlab/success_rate_110.csv", success_rate.reshape(-1, success_rate.shape[-1]), delimiter=" ")

# success_rate = np.zeros((10, 45, 3))
# file_count = 0
# if len(sr_001_files) != 0:
#     for f in sr_001_files:
#         count = 0
#         for event in summary_iterator(join(root, 'sr_001', f)):
#             for value in event.summary.value:
#                 if count >= 45:
#                     continue
#                 if "test/success_rate_medium" in value.tag:
#                     success_rate[file_count, count, 0] = event.step
#                     success_rate[file_count, count, 1] = value.simple_value
#                     count += 1
#                 elif "test/success_rate_hard" in value.tag:
#                     success_rate[file_count, count, 2] = value.simple_value
#         file_count += 1
# np.savetxt("./matlab/success_rate_001.csv", success_rate.reshape(-1, success_rate.shape[-1]), delimiter=" ")

# success_rate = np.zeros((50, 4))
# count = 0
# for event in summary_iterator('./saved/RecurrentPPO_EVAL_3/events.out.tfevents.1695585547.hyyu.285565.0'):
#     for value in event.summary.value:
#       if "test/success_rate_easy" in value.tag:
#         success_rate[count, 0] = event.step
#         success_rate[count, 1] = value.simple_value
#       elif "test/success_rate_medium" in value.tag:
#         success_rate[count, 2] = value.simple_value
#         count += 1
#       elif "test/success_rate_hard" in value.tag:
#         success_rate[count, 3] = value.simple_value

# for event in summary_iterator('./saved/RecurrentPPO_EVAL_4/events.out.tfevents.1695628603.hyyu.13301.0'):
#     for value in event.summary.value:
#       if "test/success_rate_easy" in value.tag:
#         success_rate[count, 0] = event.step
#         success_rate[count, 1] = value.simple_value
#       elif "test/success_rate_medium" in value.tag:
#         success_rate[count, 2] = value.simple_value
#         count += 1
#       elif "test/success_rate_hard" in value.tag:
#         success_rate[count, 3] = value.simple_value
# np.savetxt("./matlab/success_rate_011.txt", success_rate, delimiter=" ")

# success_rate = np.zeros((50, 4))
# count = 0
# for event in summary_iterator('./saved/RecurrentPPO_EVAL_5/events.out.tfevents.1695645951.hyyu.30688.0'):
#     for value in event.summary.value:
#       if "test/success_rate_easy" in value.tag:
#         success_rate[count, 0] = event.step
#         success_rate[count, 1] = value.simple_value
#       elif "test/success_rate_medium" in value.tag:
#         success_rate[count, 2] = value.simple_value
#         count += 1
#       elif "test/success_rate_hard" in value.tag:
#         success_rate[count, 3] = value.simple_value
# np.savetxt("./matlab/success_rate_101.txt", success_rate, delimiter=" ")

# success_rate = np.zeros((20, 4))
# count = 0
# for event in summary_iterator('./saved/RecurrentPPO_EVAL_7/events.out.tfevents.1695675920.robot-Lenovo-Legion-R7000-2020.203791.0'):
#     for value in event.summary.value:
#       if "test/success_rate_easy" in value.tag:
#         success_rate[count, 0] = event.step
#         success_rate[count, 1] = value.simple_value
#       elif "test/success_rate_medium" in value.tag:
#         success_rate[count, 2] = value.simple_value
#         count += 1
#       elif "test/success_rate_hard" in value.tag:
#         success_rate[count, 3] = value.simple_value
# np.savetxt("./matlab/success_rate_001.txt", success_rate, delimiter=" ")