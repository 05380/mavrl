""" Some data loading utilities """
from bisect import bisect
from os import listdir, path
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import rosbag
from cv_bridge import CvBridge
import cv2
import random

class _RolloutDataset(torch.utils.data.Dataset): # pylint: disable=too-few-public-methods
    def __init__(self, root, transform, buffer_size=400, train=True): # pylint: disable=too-many-arguments
        self._transform = transform

        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd)) if ssd.endswith('.npz')
        ]
        random.shuffle(self._files)
        # print(self._files)
        if train:
            self._files = self._files[0:150]
        else:
            self._files = self._files[-10:]

        self._buffer = None
        self._buffer_index = 0
        self._file_index = 0
        self._buffer_size = buffer_size
        self._cum_size = 0

    def load_next_buffer(self):
        """ Loads next buffer """
        seg_size = self._buffer_size
        self._buffer = []
        if self._file_index >= len(self._files):
            self._file_index = 0
        for file_index in range(self._file_index, len(self._files)):
            fname = self._files[file_index]
            with np.load(fname, allow_pickle=True) as data:
                data_count = data['observations'].item()['image'].shape[0]
                if self._buffer_index + seg_size < data_count:
                    data_seg = data['observations'].item()['image'][self._buffer_index:(self._buffer_index + seg_size)]
                    self._buffer += [sub_data for sub_data in data_seg]
                    self._buffer_index += seg_size
                    seg_size = self._buffer_size
                    self._cum_size += 1
                    break
                else:
                    data_seg= data['observations'].item()['image'][self._buffer_index:]
                    self._buffer += [sub_data for sub_data in data_seg]
                    seg_size = self._buffer_size - (data_count - self._buffer_index)
                    self._buffer_index = 0
                    self._file_index += 1

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return len(self._buffer)

    def __getitem__(self, i):
        if self._transform:
            return self._transform(self._buffer[i].squeeze().astype(np.uint8))
        return self._buffer[i]

class RosbagDataset(torch.utils.data.Dataset):
    def __init__(self, rosbag_folder, image_topic, buffer_size=10, transform=None, train=True):

        self.image_topic = image_topic
        self.transform = transform
        self.bag_paths = [
            join(rosbag_folder, sd, ssd)
            for sd in listdir(rosbag_folder) if isdir(join(rosbag_folder, sd))
            for ssd in listdir(join(rosbag_folder, sd))]
        if train:
            self.bag_paths = self.bag_paths[:-5]
        else:
            self.bag_paths = self.bag_paths[-5:]

        self.bridge = CvBridge()
        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size
        # rospy.init_node('rosbag_dataset', anonymous=True)

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self.bag_paths[self._buffer_index:(self._buffer_index + self._buffer_size)]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self.bag_paths)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with rosbag.Bag(f, "r") as bag:
                current_data = []
                for _, msg, _ in bag.read_messages(topics=[self.image_topic]):
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") / 1000.0
                        cv_image = (np.minimum(cv_image, 12.0)) / 12.0 * 255.0
                        shape = cv_image.shape
                        cv_image = cv_image[:, int((shape[1]-shape[0])/2 - 1) : int((shape[1]+shape[0])/2 - 1)]
                        dim = (256, 256)
                        cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
                        # convert to uint8
                        cv_image = cv_image.astype(np.uint8)
                        if self.transform:
                            cv_image = self.transform(cv_image)
                        current_data += [cv_image]
                    except Exception as e:
                        print(f"Error extracting image: {str(e)}")
                self._cum_size += [self._cum_size[-1] + len(current_data)]
                self._buffer.append(current_data)
            pbar.update(1)
        pbar.close()
    
    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]
                
    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        
        return data[seq_index]
    
class RosbagSequenceDataset(RosbagDataset):
    def __getitem__(self, index):
        return np.array(self._buffer[index]).squeeze()
    
    def __len__(self):
        return len(self._buffer)

class RolloutLSTMSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root, device, train=True):
        self._files = [
            join(root, sd, ssd)
            for sd in listdir(root) if isdir(join(root, sd))
            for ssd in listdir(join(root, sd)) if ssd.endswith('.npz')
        ]
        if train:
            self._files = self._files[:-10]
        else:
            self._files = self._files[-10:]

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = 20
        self.device = device

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = 0

        for f in self._buffer_fnames:
            with np.load(f, allow_pickle=True) as data:
                # if v is also a dict, then we need to copy each element of v, else just copy v
                self._buffer += [{k: np.copy(v) if type(v) is not dict else {kk: np.copy(vv) for kk, vv in v.items()} for k, v in data.items()}]
                self._cum_size += 1

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return len(self._buffer)
    
    def __getitem__(self, i):
        # print(self._buffer[i].keys())
        observations = {key: torch.tensor(obs, dtype=torch.float32, device=self.device) for (key, obs) in self._buffer[i]['observations'].item().items()}
        lstm_states = (torch.tensor(self._buffer[i]['lstm_states'][0], dtype=torch.float32, device=self.device), 
                       torch.tensor(self._buffer[i]['lstm_states'][1], dtype=torch.float32, device=self.device))
        episode_starts = torch.tensor(self._buffer[i]['episode_starts'], dtype=torch.float32, device=self.device)
        return observations, lstm_states, episode_starts
