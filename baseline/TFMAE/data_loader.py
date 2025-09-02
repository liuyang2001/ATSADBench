import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
# data_loader.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of samples in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SWaT_train.npy", allow_pickle=True)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SWaT_test.npy", allow_pickle=True)
        self.test = self.scaler.transform(test_data)
        self.train = data[:(int)(len(data) * 0.8)]
        self.val = data[(int)(len(data) * 0.8):]
        self.test_labels = np.load(data_path + "/SWaT_test_label.npy", allow_pickle=True)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        else:
            return (self.val.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else:
            return np.float32(self.val[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class mySegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", dataset="M-IL-FVA"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.dataset = dataset
        self.data_path = data_path
        self.scaler = StandardScaler()

        if self.dataset in ["M-IL-FVA", "M-IL-CDA", "M-IL-TVDA", "M-OL-FVA", "M-OL-CDA", "M-OL-TVDA"]:
            train_file = os.path.join(self.data_path, "first_to_sixth_shared_train_data.xlsx")
            self.target = ['Variable 1','Variable 2','Variable 3','Variable 4','Variable 5', 'Variable 6', 'Variable 7', 'Variable 8','Variable 9','Variable 10','Variable 11','Variable 12','Variable 13','Variable 14','Variable 15','Variable 16','Variable 17','Variable 18','Variable 19','Variable 20','Variable 21','Variable 22','Variable 23','Variable 24','Variable 25','Variable 26','Variable 27']  
        else:  # ["U-CDA", "U-FVA", "U-TVDA"]
            train_file = os.path.join(self.data_path, "seventh_to_ninth_shared_train_data.xlsx")
            self.target = 'Data' 

        test_file = os.path.join(self.data_path, f"{self.dataset}_test_data.xlsx")

        train_data = pd.read_excel(train_file)
        train_values = train_data[self.target] if isinstance(self.target, str) else train_data[self.target]
        train_values = train_values.values
        train_values = np.nan_to_num(train_values)

        if train_values.ndim == 1:
            train_values = train_values.reshape(-1, 1)  

        self.scaler.fit(train_values)
        train_values = self.scaler.transform(train_values)

        test_data = pd.read_excel(test_file)
        test_values = test_data[self.target] if isinstance(self.target, str) else test_data[self.target]
        test_values = test_values.values
        test_values = np.nan_to_num(test_values)

        if test_values.ndim == 1:
            test_values = test_values.reshape(-1, 1)  

        self.test = self.scaler.transform(test_values)

        self.train = train_values[:(int)(len(train_values) * 0.8)]
        self.val = train_values[(int)(len(train_values) * 0.8):]

        self.train_segment_boundaries = train_data['Segment_Boundary'].values[:(int)(len(train_values) * 0.8)]
        self.val_segment_boundaries = train_data['Segment_Boundary'].values[(int)(len(train_values) * 0.8):]

        self.test_labels = test_data['Label'].values
        self.test_segment_boundaries = test_data['Segment_Boundary'].values

        if self.mode == "train":
            self.data = self.train
            self.segment_boundaries = self.train_segment_boundaries
        elif self.mode == "val":
            self.data = self.val
            self.segment_boundaries = self.val_segment_boundaries
        else:
            self.data = self.test
            self.segment_boundaries = self.test_segment_boundaries

        data_len = self.data.shape[0]
        self.valid_indices = []
        i = 0
        while i <= data_len - self.win_size:
            window_boundaries = self.segment_boundaries[i:i + self.win_size]
            if 1 in window_boundaries[:-1]:
                next_boundary_idx = i + np.where(window_boundaries == 1)[0][-1] + 1
                i = next_boundary_idx
            else:
                self.valid_indices.append(i)
                i += self.step if self.mode in ["train", "val"] else self.win_size

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        actual_index = self.valid_indices[index]
        input_data = np.float32(self.data[actual_index:actual_index + self.win_size])
        if self.mode in ["train", "val"]:
            dummy_labels = np.zeros(self.win_size, dtype=np.float32)
            return input_data, dummy_labels
        else:
            labels = np.float32(self.test_labels[actual_index:actual_index + self.win_size])
            return input_data, labels

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    custom_datasets = [
        "M-IL-FVA", "M-IL-CDA", "M-IL-TVDA",
        "M-OL-FVA", "M-OL-CDA", "M-OL-TVDA",
        "U-CDA", "U-FVA", "U-TVDA"
    ]

    if dataset in custom_datasets:
        dataset = mySegLoader(data_path, win_size, step, mode, dataset)
    elif dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'SWaT':
        dataset = SWaTSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets are {custom_datasets + ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']}")

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8)
    return data_loader