#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from scipy import stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MTL loss (Base-class)
class MTLLoss(nn.Module):
    def __init__(self, loss_reg, loss_clsf, loss_wes, strategy):
        super().__init__()
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.wes = loss_wes
        self.strategy = strategy

    def _is_rainrate_epoch(self, epoch):
        if epoch >= self.strategy["rainrate_start"] and epoch < self.strategy["rainrate_end"]:
            return True
        else:
            return False
    
    def _is_rainmask_epoch(self, epoch):
        if epoch >= self.strategy["rainmask_start"] and epoch < self.strategy["rainmask_end"]:
            return True
        else:
            return False
    
    def _is_cloudwater_epoch(self, epoch):
        if epoch >= self.strategy["cloudwater_start"] and epoch < self.strategy["cloudwater_end"]:
            return True
        else:
            return False
    
    def _is_cloudice_epoch(self, epoch):
        if epoch >= self.strategy["cloudice_start"] and epoch < self.strategy["cloudice_end"]:
            return True
        else:
            return False
    
    def _is_mix_epoch(self, epoch):
        if epoch >= self.strategy["mix_start"] and epoch < self.strategy["mix_end"]:
            return True
        else:
            return False

    def _is_weight_mix_epoch(self, epoch):
        if epoch >= self.strategy["weighted_mix_start"] and epoch < self.strategy["weighted_mix_end"]:
            return True
        else:
            return False

    def _is_wes_epoch(self, epoch):
        if epoch >= self.strategy["wes_start"] and epoch < self.strategy["wes_end"]:
            return True
        else:
            return False


class MTLLoss_RM(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, loss_wes, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, loss_wes, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.loss_wes = loss_wes
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def forward(self, p_rainrate, p_rainmask, l_rainrate, l_rainmask, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask 
            loss_type = "mix"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss

    def get_lossbreak(self, p_rainrate, p_rainmask, l_rainrate, l_rainmask, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            return [epoch, loss_rainrate.item(), loss_rainmask.item()]
        else:
            return [epoch, np.nan, np.nan]



class MTLLoss_CW(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, loss_wes, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, loss_wes, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.loss_wes = loss_wes
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def forward(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudwater_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_type = "cloudwater"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater
            loss_type = "mix"
        elif self._is_wes_epoch(epoch):
            batch_loss = self.wes(p_rainrate, l_rainrate)
            loss_type = "wes"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss
    
    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan]


class MTLLoss_CI(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, loss_wes, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, loss_wes, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.loss_wes = loss_wes
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def forward(self, p_rainrate, p_rainmask, p_cloudice, l_rainrate, l_rainmask, l_cloudice, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudwater_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudice, l_cloudice)
            loss_type = "cloudice"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudice
            loss_type = "mix"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss
    
    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudice, l_rainrate, l_rainmask, l_cloudice, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudice.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan]



class MTLLoss_CWCI(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, loss_wes, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, loss_wes, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.loss_wes = loss_wes
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def forward(self, p_rainrate, p_rainmask, p_cloudwater, p_cloudice, l_rainrate, l_rainmask, l_cloudwater, l_cloudice, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudwater_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_type = "cloudwater"
        elif self._is_cloudice_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudice, l_cloudice)
            loss_type = "cloudice"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater + self.loss_weight[3]*loss_cloudice
            loss_type = "mix"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss
    
    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudwater, p_cloudice, l_rainrate, l_rainmask, l_cloudwater, l_cloudice, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_cloudice = self.loss_reg(p_cloudice, l_cloudice)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater.item(), loss_cloudice.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan, np.nan]


class MTLLoss_CW_AccordanceWeighting(MTLLoss):
    def __init__(self, loss_reg, loss_clsf, loss_wes, strategy, loss_weight):
        super().__init__(loss_reg, loss_clsf, loss_wes, strategy)
        self.loss_reg = loss_reg
        self.loss_clsf  = loss_clsf
        self.loss_wes = loss_wes
        self.strategy = strategy
        self.loss_weight = loss_weight
    
    def weigted_loss_reg(self, p_rainrate, p_cloudwater, l_cloudwater):
        squared_error = (p_cloudwater - l_cloudwater)**2
        p_rainrate4weiting = p_rainrate.detach()
        weigting_factor = self.weighting_func(p_rainrate4weiting, a=1, b=60)
        mean_weigted_loss = torch.mean(squared_error * weigting_factor)
        return mean_weigted_loss
    

    def weighting_func(self, x, a=1, b=60):
        # x = torch.round(x, decimals=3)
        """
        a = line-smoothing parameter
        b = threshold for weighting
        """
        y = 1/(1+torch.exp(-(x-b)*a))
        return -1*y + 1
    
    def forward(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_rainrate_epoch(epoch):
            batch_loss = self.loss_reg(p_rainrate, l_rainrate)
            loss_type = "rainrate"
        elif self._is_rainmask_epoch(epoch):
            batch_loss = self.loss_clsf(p_rainmask, l_rainmask)
            loss_type = "rainmask"
        elif self._is_cloudwater_epoch(epoch):
            batch_loss = self.loss_reg(p_cloudwater, l_cloudwater)
            loss_type = "cloudwater"
        elif self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater
            loss_type = "mix"
        elif self._is_weight_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater_weigted = self.weigted_loss_reg(p_rainrate, p_cloudwater, l_cloudwater)
            batch_loss = self.loss_weight[0]*loss_rainrate + self.loss_weight[1]*loss_rainmask + self.loss_weight[2]*loss_cloudwater_weigted
            loss_type = "weight_mix"
        elif self._is_wes_epoch(epoch):
            batch_loss = self.wes(p_rainrate, l_rainrate)
            loss_type = "wes"
        else:
            raise RuntimeError("undefined epoch num!")
        return batch_loss
    
    def get_lossbreak(self, p_rainrate, p_rainmask, p_cloudwater, l_rainrate, l_rainmask, l_cloudwater, epoch):
        if self._is_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater = self.loss_reg(p_cloudwater, l_cloudwater)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater.item()]
        elif self._is_weight_mix_epoch(epoch):
            loss_rainrate = self.loss_reg(p_rainrate, l_rainrate)
            loss_rainmask = self.loss_clsf(p_rainmask, l_rainmask)
            loss_cloudwater_weigted = self.weigted_loss_reg(p_rainrate, p_cloudwater, l_cloudwater)
            return [epoch, loss_rainrate.item(), loss_rainmask.item(), loss_cloudwater_weigted.item()]
        else:
            return [epoch, np.nan, np.nan, np.nan]


class WES(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def label_pdf(self, x):
        """
        Probability density function estimated in pre-training.
        """
        est_a, est_loc, est_scale = 0.06444291424643259, -1.1328205299926424e-27, 1.5376362609160314
        y = stats.gamma.pdf(x, est_a, est_loc, est_scale)
        return y

    def error_factor(self, y):
        """
        min, max parameter computed in priiri dataset.
        """
        y_max = 107.2185
        y_min = 0
        fx_max =1.0500746950269468e+24
        
        y = y.cpu().numpy()
        c = 1 / (y_max - y_min)
        estimated_pdf_values = self.label_pdf(y)
        error_fac = ((self.beta - c) * ((-1 * estimated_pdf_values / fx_max) + 1)) + c
        error_fac = torch.tensor(error_fac).to(device)
        return error_fac

    def forward(self, output, target):
        loss = torch.mean((0.5*(output - target)**2) * self.error_factor(target))
        return loss
