#!/usr/bin/env python
# coding: utf-8
"""
Evaluate a model.
"""

import os
import glob
from data import *
from util import *
from model import *
from loss import *
from train import *
from eval import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Experiment name
exp_name = "exp_name"

print(exp_name, "Eval Start...")

path_test = "../data/dataset_112_full/test/**/*.npy"
prep_method = "norm" # norm or min_max
batch_size = 50

path_test = get_path(path_test, n_sample=None)
test_dataset = Mydataset(path_test, preprocess=prep_method)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True,
                                        worker_init_fn=worker_init_fn
                                        )


# Single-task (PERSIANN-like model) -------------------------------
print("Start. Single-task training...")
model = PersiannCNN().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN", input_type="IR_WV", MTL=False)

model = PersiannCW().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN_CW", input_type="CW", MTL=False)

# Multi-task-Multi-input -------------------------------
model = PersiannMTLMultiInput_CW().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW", input_type="CW", MTL=True, MultiInput=True)

model = PersiannMTLMultiInput_CW().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Weighting", input_type="CW", MTL=True, MultiInput=True)

model = PersiannMTLMultiInput_CW().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Strategy", input_type="CW", MTL=True, MultiInput=True)

model = PersiannMTLMultiInput_CW().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Strategy(WES)", input_type="CW", MTL=True, MultiInput=True)

model = PersiannMTLMultiInput_CW().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Weighting_Strategy", input_type="CW", MTL=True, MultiInput=True)

model = PersiannMTLMultiInput_CW().to(device)
compute_evaluation(model, device, path_test, test_loader, save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Weighting_Strategy(WES)", input_type="CW", MTL=True, MultiInput=True)
