#!/usr/bin/env python
# coding: utf-8

import glob
import numpy as np
import torch
from tqdm import tqdm


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, path_data,  preprocess="norm"):
        self.path_data = path_data
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.path_data)
   
    def normalize_feature(self, x):
        mean = [2.754951e+02, 2.754951e+02, 2.757481e+02, 2.757481e+02, 7.385361e-02, 7.385361e-02, 9.049441e-02, 9.049441e-02]
        std = [1.160923e-01, 1.160923e-01, 2.856966e-01, 2.856966e-01, 1.183280e-01, 1.183280e-01, 1.687981e-01, 1.687981e-01]
        for i in range(x.shape[0]):
            x[i] = (x[i]-mean[i])/std[i]
        return x
    
    def min_max(self, x):
        # min-maxの設定の仕方を工夫（極値まで含みすぎか, Precpデータから閾値を設定）？
        x_min = [2.750782e+02, 2.750782e+02, 2.750723e+02, 2.750723e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
        x_max = [2.758907e+02, 2.758907e+02, 2.764080e+02, 2.764080e+02, 2.150606e+00, 2.150606e+00, 2.339565e+00, 2.339565e+00]
        for i in range(x.shape[0]):
            x[i] = (x[i] - (x_min[i]))/(x_max[i] - x_min[i])
        return x

    def __getitem__(self, index):
        feature = np.load(self.path_data[index])
        y = feature[0, :, :]
        x = feature[1:,  :, :]
        # Preprocess
        if self.preprocess == "norm":
            x = self.normalize_feature(x)
        elif self.preprocess == "min_max":
            x = self.min_max(x)
        y = torch.tensor(y)
        x = torch.tensor(x)
        return y.unsqueeze(0), x

def get_path(path_feature, n_sample=None, screening=None):
    path_feature = glob.glob(path_feature, recursive=True)
    if n_sample is not None:
        np.random.shuffle(path_feature)
        path_feature = path_feature[:n_sample]
    if screening is not None:
        print("Data screening...")
        path_feature = data_screening(path_feature, screening)
    path_feature.sort()
    return path_feature


def data_screening(path_feature, matchin_ratio = 0.5):
    # Mask threshhold
    wv_th = 275.5
    cw_th = 0.025
    ci_th = 0.01

    cw_match, ci_match = [], []
    for p in tqdm(path_feature):
        f = np.load(p)
        # Get data for the validation
        wv = f[2, :, :]
        cw = f[6, :, :]
        ci = f[8, :, :]
        wv_mask = (wv <= wv_th)*1
        cw_mask = (cw >= cw_th)*1
        ci_mask = (ci >= ci_th)*1
        cw_match.append((wv_mask == cw_mask).sum()/len(wv_mask.reshape(-1)))
        ci_match.append((wv_mask == ci_mask).sum()/len(wv_mask.reshape(-1)))

    cw_match = np.array(cw_match)
    ci_match = np.array(ci_match)

    screen = (cw_match >= matchin_ratio) & (ci_match >= matchin_ratio)
    screened_path = np.array(path_feature)[screen]
    return screened_path.tolist()


def train_val_shuffle(path_train, path_val):
    """
    Shuffle and 80% and 20% split.
    - 2012/6-8, 2013/6: Tain
    - 2013/7: Val
    """
    path_train_val = path_train + path_val
    np.random.shuffle(path_train_val)

    n = len(path_train_val)
    train_boarder = int(n * 0.8)
    path_train = path_train_val[:train_boarder]
    path_val = path_train_val[train_boarder:]
    return path_train, path_val

def prepare_dataset(path_train, path_val, path_test, n_sample, prep_method, screening):
    path_train = get_path(path_train, n_sample=n_sample, screening=screening)
    path_val = get_path(path_val, n_sample=n_sample, screening=screening)
    path_test = get_path(path_test, n_sample=n_sample)
    path_train, path_val = train_val_shuffle(path_train, path_val)
    train_dataset = Mydataset(path_train, preprocess=prep_method)
    val_dataset = Mydataset(path_val, preprocess=prep_method)
    test_dataset = Mydataset(path_test, preprocess=prep_method)
    return train_dataset, val_dataset, test_dataset

