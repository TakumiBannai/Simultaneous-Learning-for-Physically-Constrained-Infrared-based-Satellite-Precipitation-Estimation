#!/usr/bin/env python
# coding: utf-8
"""
Train a model.
python3 main.py test_full
"""

from data import *
from util import *
from model import *
from loss import *
from train import *
from eval import *
import sys

args = sys.argv
exp_name = args[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fix_seed(42)

# Experiment name
print(exp_name, "Experiment Start...")

# Parameters
num_epoch = 150
num_batch_size = 48
loss_weight = [1, 1, 1, 1] # Weighting factor for mixed loss (rainrate, rainmask, cloud water, cloud ice)
early_stop = True
stop_after = 100
loss_break = False
screening = None

# Path to Dataset
path_train = "../data/dataset_112_10per/train/**/*.npy"
path_val = "../data/dataset_112_10per/val/**/*.npy"
path_test = "../data/dataset_112_10per/test/**/*.npy"

train_dataset, val_dataset, test_dataset = prepare_dataset(path_train, 
                                                           path_val,
                                                           path_test,
                                                           n_sample = None,
                                                           prep_method = "norm",
                                                           screening = screening)
print("Data size: ", len(train_dataset), len(val_dataset), len(test_dataset))


# --------------------------------------------------------------
print("Start. Model training...")

# Single-task (PERSIANN-like model) ------------
# No-Strategy
strategy = {
    "rainmask_start":0, "rainmask_end":20,
    "mix_start":20, "mix_end":150,
    
    "rainrate_start":230, "rainrate_end":250,
    "weighted_mix_start":150, "weighted_mix_end":200, 
    "cloudwater_start":210, "cloudwater_end":220,
    "cloudice_start":220, "cloudice_end":230
    }

print("Start. Model training...")
model = PersiannCNN().to(device)
run_PERSIANN(model, train_dataset, val_dataset, device,
             cloud_type="IR_WV", batch_size = num_batch_size, epoch = num_epoch,
             opt = "Adam", lr=0.0001, 
             save_dir = f"./output/{exp_name}/PERSIAN",
             early_stop=early_stop, stop_after=stop_after)

model = PersiannCW().to(device)
run_PERSIANN(model, train_dataset, val_dataset, device,
             cloud_type="CW", batch_size = num_batch_size, epoch = num_epoch,
             opt = "Adam", lr=0.0001, 
             save_dir = f"./output/{exp_name}/PERSIAN_CW",
             early_stop=early_stop, stop_after=stop_after)


# Multi-task-Multi-input -------------------------------
model = PersiannMTLMultiInput_CW().to(device)
MTLloss = MTLLoss_CW
run_PERSIANN_MTL_MultiInput(model, train_dataset, val_dataset, device,
                            MTLloss, strategy, loss_weight, 
                            cloud_type="CW", batch_size = num_batch_size, epoch = num_epoch,
                            opt = "Adam", lr=0.0001, 
                            save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW",
                            loss_break=loss_break,
                            early_stop=early_stop, stop_after=stop_after)


# No-Strategy (Weigted loss)
strategy = {
    "rainmask_start":0, "rainmask_end":20,
    "mix_start":20, "mix_end":80,
    "weighted_mix_start":80, "weighted_mix_end":150,
    "rainrate_start":150, "rainrate_end":200,
    
    "cloudwater_start":210, "cloudwater_end":220,
    "cloudice_start":220, "cloudice_end":230
    }


model = PersiannMTLMultiInput_CW().to(device)
MTLloss = MTLLoss_CW_AccordanceWeighting
run_PERSIANN_MTL_MultiInput(model, train_dataset, val_dataset, device,
                            MTLloss, strategy, loss_weight, 
                            cloud_type="CW", batch_size = num_batch_size, epoch = num_epoch,
                            opt = "Adam", lr=0.0001, 
                            save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Weighting",
                            loss_break=loss_break,
                            early_stop=early_stop, stop_after=stop_after)

# Strategy
strategy = {
    "rainmask_start":0, "rainmask_end":20,
    "mix_start":20, "mix_end":60,
    "rainrate_start":60, "rainrate_end":150,
    
    "weighted_mix_start":150, "weighted_mix_end":200, 
    "cloudwater_start":210, "cloudwater_end":220,
    "cloudice_start":220, "cloudice_end":230
    }

model = PersiannMTLMultiInput_CW().to(device)
MTLloss = MTLLoss_CW
run_PERSIANN_MTL_MultiInput(model, train_dataset, val_dataset, device,
                            MTLloss, strategy, loss_weight, 
                            cloud_type="CW", batch_size = num_batch_size, epoch = num_epoch,
                            opt = "Adam", lr=0.0001, 
                            save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Strategy",
                            loss_break=loss_break,
                            early_stop=early_stop, stop_after=stop_after)


# Strategy & Weigted loss
strategy = {
    "rainmask_start":0, "rainmask_end":20,
    "mix_start":20, "mix_end":50,
    "weighted_mix_start":50, "weighted_mix_end":80,
    "rainrate_start":80, "rainrate_end":150,
    
    "cloudwater_start":210, "cloudwater_end":220,
    "cloudice_start":220, "cloudice_end":230
    }
model = PersiannMTLMultiInput_CW().to(device)
MTLloss = MTLLoss_CW_AccordanceWeighting
run_PERSIANN_MTL_MultiInput(model, train_dataset, val_dataset, device,
                            MTLloss, strategy, loss_weight, 
                            cloud_type="CW", batch_size = num_batch_size, epoch = num_epoch,
                            opt = "Adam", lr=0.0001, 
                            save_dir = f"./output/{exp_name}/PERSIAN_MTMI_CW_Weighting_Strategy",
                            loss_break=loss_break,
                            early_stop=early_stop, stop_after=stop_after)


print("Done. Model training...")

# --------------------------------------------------------------
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

print("Done. Model evaluation...")
