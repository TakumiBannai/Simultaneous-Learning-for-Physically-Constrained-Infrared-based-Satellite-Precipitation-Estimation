#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from util import *
from eval import *
from loss import *

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def save_training_result(device, model, train_loss, val_loss, save_dir):
        # Trained model
        torch.save(model.state_dict(), f"{save_dir}/TrainedModel")
        # Train/Val loss decreasing
        plt.plot(train_loss, label="train")
        plt.plot(val_loss, label="val")
        plt.ylim(2.3, 3.8)
        plt.legend()
        np.save(f"{save_dir}/TrainBatchLoss.npy", train_loss)
        np.save(f"{save_dir}/ValBatchLoss.npy", val_loss)
        plt.savefig(f"{save_dir}/TrainValLoss.png")
        plt.close()

def run_PERSIANN(model, train_dataset, val_dataset, device,
                 cloud_type="IR_WV", batch_size = 48,
                 epoch = 50, opt = "Adam", lr=0.0001,
                 save_dir = "./output/test",
                 early_stop=False, stop_after=0):
    # Config.
    criterion = nn.MSELoss()

    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    elif opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr)

    print("Train Config.")
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = open(f"{save_dir}/TrainConfig.txt","w")
    
    print("loss func: ", criterion)
    print("Optimizer: ", optimizer)
    print(f"cloud_type: {cloud_type}")
    print(f"batch_size: {batch_size}")
    print(f"epoch: {epoch}")
    print(f"Learning rate: {lr}")
    print(f"save_dir: {save_dir}")
    start = time.time()

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn
                                             )
    print("dataloader(train/val): ", len(train_loader.dataset), len(val_loader.dataset))

    # Model training --------------
    print("Model train start...")
    train_loss, val_loss = [], []
    earlystopping = EarlyStopping(patience=8, verbose=True, path=f"{save_dir}/TrainedModel")

    for i in tqdm(range(epoch)):
        # Train loop
        model.train()
        train_batch_loss = []
        for y, x  in train_loader:
            y, x = y.to(device).float(), x.to(device).float()
            wv = x[:, 0:2, :, :]
            ir = x[:, 2:4, :, :]
            cw = x[:, 4:6, :, :]
            ci = x[:, 6:8, :, :]
            optimizer.zero_grad()
            if cloud_type == "IR_WV":
                output = model(ir, wv)
            elif cloud_type == "CW":
                output = model(ir, wv, cw)
            elif cloud_type == "CI":
                output = model(ir, wv, ci)
            elif cloud_type == "CWCI":
                output = model(ir, wv, cw, ci)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())
        # val loop
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for y, x in val_loader:
                y, x = y.to(device).float(), x.to(device).float()
                wv = x[:, 0:2, :, :]
                ir = x[:, 2:4, :, :]
                cw = x[:, 4:6, :, :]
                ci = x[:, 6:8, :, :]
                if cloud_type == "IR_WV":
                    output = model(ir, wv)
                elif cloud_type == "CW":
                    output = model(ir, wv, cw)
                elif cloud_type == "CI":
                    output = model(ir, wv, ci)
                elif cloud_type == "CWCI":
                    output = model(ir, wv, cw, ci)
                loss = criterion(output, y)
                val_batch_loss.append(loss.item())
        # Collect loss
        train_loss.append(np.mean(train_batch_loss))
        val_loss.append(np.mean(val_batch_loss))
        if i % 5 == 0:
            print(i, "Train loss: {a:.3f}, Val loss: {b:.3f}".format(
                a=train_loss[-1], b=val_loss[-1]))
        
        # Early-stopping judgement
        if early_stop is True:
            if i >= stop_after:
                earlystopping(val_loss[-1], model)
                if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
                    print(f"Early Stopping: iter. {i}")
                    break

    print("Model train done...")
    elapsed_time = time.time() - start
    print(f"{elapsed_time:.2f}", ' sec.')

    # Save the result -----------------
    save_training_result(device, model, train_loss, val_loss, save_dir)

    # Close
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    

def run_PERSIANN_MTL(model, train_dataset, val_dataset, device,
                     MTLloss, strategy,loss_weight, 
                     cloud_type="IR_WV", batch_size = 48, epoch = 50, opt = "Adam", lr=0.0001, 
                     save_dir = "./output/test",
                     loss_break=False,
                     early_stop=False, stop_after=0):
    # Config.
    criterion = MTLloss(loss_reg=nn.MSELoss(), loss_clsf=nn.BCELoss(), loss_wes=WES(beta=3),
                        strategy=strategy, loss_weight=loss_weight)

    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    elif opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr)

    print("Train Config.")
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = open(f"{save_dir}/TrainConfig.txt","w")
    
    print("loss func: ", criterion)
    print("Optimizer: ", optimizer)
    print(f"cloud_type: {cloud_type}")
    print(f"batch_size: {batch_size}")
    print(f"epoch: {epoch}")
    print(f"Learning rate: {lr}")
    print(f"save_dir: {save_dir}")
    print(f"{strategy}")
    start = time.time()

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn
                                            )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn
                                            )
    print("dataloader(train/val): ", len(train_loader.dataset), len(val_loader.dataset))

    # Model training --------------
    print("Model train start...")
    train_loss, val_loss = [], []
    trainloss_break = []
    earlystopping = EarlyStopping(patience=8, verbose=True, path=f"{save_dir}/TrainedModel")

    for i in tqdm(range(epoch)):
        # Train loop
        model.train()
        train_batch_loss = []
        trainloss_break_batch = []
        for y, x  in train_loader:
            y, x = y.to(device).float(), x.to(device).float()
            y_mask = ((y >= 0.1)*1 ).float() # Rain-mask
            wv = x[:, 0:2, :, :]
            ir = x[:, 2:4, :, :]
            cw = x[:, 4:5, :, :] # Get Only Time_t
            ci = x[:, 6:7, :, :] # Get Only Time_t
            optimizer.zero_grad()
            if cloud_type == "IR_WV":
                p_rain, p_mask = model(ir, wv)
                loss = criterion(p_rain, p_mask, y, y_mask, i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, y, y_mask, i)
                    trainloss_break_batch.append(b_loss)
                
            elif cloud_type == "CW":
                p_rain, p_mask, p_cloudwater = model(ir, wv)
                loss = criterion(p_rain, p_mask, p_cloudwater, y, y_mask, cw, i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, p_cloudwater, y, y_mask, cw, i)
                    trainloss_break_batch.append(b_loss)

            elif cloud_type == "CI":
                p_rain, p_mask, p_cloudice = model(ir, wv)
                loss = criterion(p_rain, p_mask, p_cloudice, y, y_mask, ci, i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, p_cloudice, y, y_mask, ci, i)
                    trainloss_break_batch.append(b_loss)

            elif cloud_type == "CWCI":
                p_rain, p_mask, p_cloudwater, p_cloudice = model(ir, wv)
                loss = criterion(p_rain, p_mask, p_cloudwater, p_cloudice, y, y_mask, cw, ci, i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, p_cloudwater, p_cloudice, y, y_mask, cw, ci, i)
                    trainloss_break_batch.append(b_loss)
            
        # val loop
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for y, x in val_loader:
                y, x = y.to(device).float(), x.to(device).float()
                y_mask = ((y >= 0.1)*1).float()# Rain-mask
                wv = x[:, 0:2, :, :]
                ir = x[:, 2:4, :, :]
                cw = x[:, 4:5, :, :]
                ci = x[:, 6:7, :, :]
                if cloud_type == "IR_WV":
                    p_rain, p_mask = model(ir, wv)
                    loss = criterion(p_rain, p_mask, y, y_mask, i)
                    val_batch_loss.append(loss.item())
                elif cloud_type == "CW":
                    p_rain, p_mask, p_cloudwater = model(ir, wv)
                    loss = criterion(p_rain, p_mask, p_cloudwater, y, y_mask, cw, i)
                    val_batch_loss.append(loss.item())
                elif cloud_type == "CI":
                    p_rain, p_mask, p_cloudice = model(ir, wv)
                    loss = criterion(p_rain, p_mask, p_cloudice, y, y_mask, ci, i)
                    val_batch_loss.append(loss.item())
                elif cloud_type == "CWCI":
                    p_rain, p_mask, p_cloudwater, p_cloudice = model(ir, wv)
                    loss = criterion(p_rain, p_mask, p_cloudwater, p_cloudice, y, y_mask, cw, ci, i)
                    val_batch_loss.append(loss.item())
        # Collect loss
        train_loss.append(np.mean(train_batch_loss))
        val_loss.append(np.mean(val_batch_loss))
        trainloss_break.append(np.mean(trainloss_break_batch, axis=0))
        
        if i % 5 == 0:
            print(i, "Train loss: {a:.3f}, Val loss: {b:.3f}".format(
                a=train_loss[-1], b=val_loss[-1]))

        # Early-stopping judgement
        if early_stop is True:
            if i >= stop_after:
                earlystopping(val_loss[-1], model)
                if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
                    print(f"Early Stopping: iter. {i}")
                    break

    print("Model train done...")
    elapsed_time = time.time() - start
    print(f"{elapsed_time:.2f}", ' sec.')

    # Save the result -----------------
    save_training_result(device, model, train_loss, val_loss, save_dir)

    if loss_break is True:
        np.save(f"{save_dir}/TrainLossBreakdown.npy", np.array(trainloss_break))

    # Close
    sys.stdout.close()
    sys.stdout = sys.__stdout__


# ERSIANN_MTL_MultiInput
def run_PERSIANN_MTL_MultiInput(model, train_dataset, val_dataset, device,
                                MTLloss, strategy, loss_weight, 
                                cloud_type="IR_WV", batch_size = 48, epoch = 50, opt = "Adam", lr=0.0001, 
                                weight_decay=0, save_dir = "./output/test",
                                loss_break=False,
                                early_stop=False, stop_after=0):

    # Config.
    criterion = MTLloss(loss_reg=nn.MSELoss(), loss_clsf=nn.BCELoss(), loss_wes=WES(beta=3),
                        strategy=strategy, loss_weight=loss_weight)
    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    elif opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = open(f"{save_dir}/TrainConfig.txt","w")
    print("Train Config.")
    print("loss func: ", criterion)
    print("Optimizer: ", optimizer)
    print(f"cloud_type: {cloud_type}")
    print(f"batch_size: {batch_size}")
    print(f"epoch: {epoch}")
    print(f"Learning rate: {lr}")
    print(f"save_dir: {save_dir}")
    print(f"{strategy}")
    start = time.time()

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn
                                            )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True,
                                            worker_init_fn=worker_init_fn
                                            )
    print("dataloader(train/val): ", len(train_loader.dataset), len(val_loader.dataset))

    # Model training --------------
    print("Model train start...")
    train_loss, val_loss = [], []
    trainloss_break = []
    earlystopping = EarlyStopping(patience=8, verbose=True, path=f"{save_dir}/TrainedModel")

    for i in tqdm(range(epoch)):
        # Train loop
        model.train()
        train_batch_loss = []
        trainloss_break_batch = []
        for y, x  in train_loader:
            y, x = y.to(device).float(), x.to(device).float()
            y_mask = ((y >= 0.1)*1 ).float() # Rain-mask
            wv = x[:, 0:2, :, :]
            ir = x[:, 2:4, :, :]
            cw = x[:, 4:6, :, :]
            ci = x[:, 6:8, :, :]
            optimizer.zero_grad()
            if cloud_type == "IR_WV":
                p_rain, p_mask = model(ir, wv)
                loss = criterion(p_rain, p_mask, y, y_mask, i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, y, y_mask, i)
                    trainloss_break_batch.append(b_loss)
                
            elif cloud_type == "CW":
                p_rain, p_mask, p_cloudwater = model(ir, wv, cw)
                loss = criterion(p_rain, p_mask, p_cloudwater, y, y_mask, cw[:,1:2], i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, p_cloudwater, y, y_mask, cw[:,1:2], i)
                    trainloss_break_batch.append(b_loss)

            elif cloud_type == "CI":
                p_rain, p_mask, p_cloudice = model(ir, wv, ci)
                loss = criterion(p_rain, p_mask, p_cloudice, y, y_mask, ci[:,1:2], i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, p_cloudice, y, y_mask, ci[:,1:2], i)
                    trainloss_break_batch.append(b_loss)

            elif cloud_type == "CWCI":
                p_rain, p_mask, p_cloudwater, p_cloudice = model(ir, wv, cw, ci)
                loss = criterion(p_rain, p_mask, p_cloudwater, p_cloudice, y, y_mask, cw[:,1:2], ci[:,1:2], i)
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())
                # Collect loss-break(if True)
                if loss_break is True:
                    b_loss = criterion.get_lossbreak(p_rain, p_mask, p_cloudwater, p_cloudice, y, y_mask, cw[:,1:2], ci[:,1:2], i)
                    trainloss_break_batch.append(b_loss)
            
        # val loop
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for y, x in val_loader:
                y, x = y.to(device).float(), x.to(device).float()
                y_mask = ((y >= 0.1)*1).float()# Rain-mask
                wv = x[:, 0:2, :, :]
                ir = x[:, 2:4, :, :]
                cw = x[:, 4:6, :, :]
                ci = x[:, 6:8, :, :]
                if cloud_type == "IR_WV":
                    p_rain, p_mask = model(ir, wv)
                    loss = criterion(p_rain, p_mask, y, y_mask, i)
                    val_batch_loss.append(loss.item())
                elif cloud_type == "CW":
                    p_rain, p_mask, p_cloudwater = model(ir, wv, cw)
                    loss = criterion(p_rain, p_mask, p_cloudwater, y, y_mask, cw[:,1:2], i)
                    val_batch_loss.append(loss.item())
                elif cloud_type == "CI":
                    p_rain, p_mask, p_cloudice = model(ir, wv, ci)
                    loss = criterion(p_rain, p_mask, p_cloudice, y, y_mask, ci[:,1:2], i)
                    val_batch_loss.append(loss.item())
                elif cloud_type == "CWCI":
                    p_rain, p_mask, p_cloudwater, p_cloudice = model(ir, wv, cw, ci)
                    loss = criterion(p_rain, p_mask, p_cloudwater, p_cloudice, y, y_mask, cw[:,1:2], ci[:,1:2], i)
                    val_batch_loss.append(loss.item())
        # Collect loss
        train_loss.append(np.mean(train_batch_loss))
        val_loss.append(np.mean(val_batch_loss))
        trainloss_break.append(np.mean(trainloss_break_batch, axis=0))
        
        if i % 5 == 0:
            print(i, "Train loss: {a:.3f}, Val loss: {b:.3f}".format(
                a=train_loss[-1], b=val_loss[-1]))

        # Early-stopping judgement
        if early_stop is True:
            if i >= stop_after:
                earlystopping(val_loss[-1], model)
                if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
                    print(f"Early Stopping: iter. {i}")
                    break

    print("Model train done...")
    elapsed_time = time.time() - start
    print(f"{elapsed_time:.2f}", ' sec.')

    # Save the result -----------------
    save_training_result(device, model, train_loss, val_loss, save_dir)
    if loss_break is True:
        np.save(f"{save_dir}/TrainLossBreakdown.npy", np.array(trainloss_break))

    # Close
    sys.stdout.close()
    sys.stdout = sys.__stdout__

