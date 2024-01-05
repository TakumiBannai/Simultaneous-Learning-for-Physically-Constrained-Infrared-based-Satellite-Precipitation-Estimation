#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
from util import *


class EvaluationIndices():
    """
    rain_th_clsf: threshold for rain-no rain classification
    raih_th_reg: threshold for evaluationf of regression
    """
    def __init__(self, pred_reg, label_reg, rain_th_clsf=0.1, rain_th_reg=0):
        self.pred_reg = pred_reg.reshape(-1)
        self.label_reg = label_reg.reshape(-1)
        self.rain_th_clsf = rain_th_clsf
        self.rain_th_reg = rain_th_reg
        self.pred_cls = (self.pred_reg >= self.rain_th_clsf)*1
        self.label_cls = (self.label_reg >= self.rain_th_clsf)*1

    # Regression
    def me(self, a, b):
        return mean_error(a, b)
    def mae(self, a, b):
        return mean_absolute_error(a, b)
    def rmse(self, a, b):
        return mean_squared_error(a, b, squared=False)
    def cc(self, a, b):
        return np.corrcoef(a, b)[0][1]
    # Classification
    def pod(self, h, m):
        return h / (h + m)
    def far(self, f, h):
        return f / (h + f)
    def bias(self, h, f, m):
        return (h + f) / (h + m)
    def ets(self, h, c, f, m):
        r = ((h + m)*(h + f)) / (h + m + f + c)
        return (h - r) / (h + m + f - r)
    def hss(self, h, c, f, m):
        nume = 2*(h * c - f * m)
        denom = ((h + m) * (m + c)) + ((h + f) * (f + c))
        return nume / denom
    def csi(self, h, f, m):
    	return h / (h + f +m)

    def evaluate(self):
        num = len(self.pred_reg)
        if self.rain_th_reg != 0:
            # Filtering by rain_th_reg
            # raining = (self.pred_reg >= self.rain_th_reg) & (self.label_reg >= self.rain_th_reg)
            raining = (self.label_reg >= self.rain_th_reg) # LableのみでFiltering
            self.pred_reg = self.pred_reg[raining]
            self.label_reg = self.label_reg[raining]
            # Dummy number for classification
            c, f, m, h = 0.25, 0.25, 0.25, 0.25
        else:
            c_matrix = confusion_matrix(self.label_cls.reshape(-1), self.pred_cls.reshape(-1))
            tn, fp, fn, tp = c_matrix.ravel()
            c, f, m, h = tn, fp, fn, tp
        return {
                "MAE": self.mae(self.pred_reg, self.label_reg),
                "RMSE": self.rmse(self.pred_reg, self.label_reg),
                "CC": self.cc(self.pred_reg, self.label_reg),
                "POD": self.pod(h, m),
                "FAR": self.far(f, h),
                "CSI": self.csi(h, f, m)
                }


def compute_evaluation(model, device, path_test, test_loader, save_dir = "./output/test",
                       input_type="IR_WV", rain_th = 0.1, MTL=False, MultiInput=False, post_process=False):
    # Load trained model
    path_saved_model = f"{save_dir}/TrainedModel"
    model.load_state_dict(torch.load(path_saved_model))
    # Get Prediction
    preds, labels = retrieve_result(device, model, test_loader, input_type, MTL, MultiInput)
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    # Evaluation
    evaluater = EvaluationIndices(preds, labels, rain_th=0.1)
    eval_test = pd.DataFrame([evaluater.evaluate()])
    print(eval_test)
    os.makedirs(f"{save_dir}/Evaluation", exist_ok=True)
    eval_test.to_csv(f"{save_dir}/Evaluation/index_eval.csv")

    # Rain-intensity interval
    eval_rain_intensity = compute_index_bin(preds, labels, out_type='df')
    eval_rain_intensity.to_csv(f"{save_dir}/Evaluation/index_eval_interval.csv")

    # Density scatter plot (Train)
    show_regression_performance_2d(labels, preds, "Test (2013/8)")
    plt.savefig(f"{save_dir}/Evaluation/DensityScatterPlot_Test.png")

    if post_process is True:
        preds_trans = liner_transformation_polynomial(preds)

        # Evaluation
        evaluater = EvaluationIndices(preds_trans, labels, rain_th=0.1)
        eval_test = pd.DataFrame([evaluater.evaluate()])
        eval_test.to_csv(f"{save_dir}/Evaluation/index_eval_PostProcess.csv")

        eval_rain_intensity = compute_index_bin(preds_trans, labels, out_type='df')
        eval_rain_intensity.to_csv(f"{save_dir}/Evaluation/index_eval_interval_PostProcess.csv")

        show_regression_performance_2d(labels, preds_trans, "Test (2013/8)")
        plt.savefig(f"{save_dir}/Evaluation/DensityScatterPlot_Test_PostProcess.png")

    # Save 1-d Preds value
    np.save(f"{save_dir}/Evaluation/preds.npy", preds)
    np.save(f"{save_dir}/Evaluation/labels.npy", labels)

    # Save preds as tiles
    path_test.sort()
    for i in tqdm(range(len(path_test))):
       new_name = path_test[i].replace("../data/", f"{save_dir}/Evaluation/")
       os.makedirs(os.path.dirname(new_name), exist_ok=True)
       np.save(new_name, preds[i])


# Compute evaluation index
def compute_index(A, B):
    A = A.reshape(-1)
    B = B.reshape(-1)
    # Number of Pixcel
    num_pix = len(A)
    # MAE
    mae = mean_absolute_error(A, B)
    # MSE
    mse = mean_squared_error(A, B, squared=True)
    # RMSE
    rmse = mean_squared_error(A, B, squared=False)
    # Corr
    corr = np.corrcoef(A, B)
    return num_pix, mae, mse, rmse, corr[0][1]

# x: label, y: pred
def show_regression_performance_2d(label, pred, title_name, vmax=1900):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    # log10
    with np.errstate(divide='ignore'):
        label = np.log10(label)
        pred = np.log10(pred)
    # plot
    plt.figure(figsize=(7, 5.5))
    plt.hist2d(label, pred, bins=(55, 55),
               vmax=vmax,
               cmap=cm.jet, range=[[0.01, 1.5], [0.01, 1.5]])
    plt.colorbar()
    plt.xlabel('Label: Precipitation ($log_{10}$[mm/h])')
    plt.ylabel('Pred: Precipitation ($log_{10}$[mm/h])')
    plt.title('{a}'.format(a=title_name))
    plt.xticks(np.arange(0, 1.75, 0.25))
    plt.yticks(np.arange(0, 1.75, 0.25))
    # daiagonal line
    ident = [0.01, 1.5]
    plt.plot(ident, ident, ls="--", lw="1.2", c="gray")

# Rain intensity class
def binning(arr_pred, arr_label, bin="weak"):
    no_rain = (0 <= arr_label) & (arr_label < 0.1)
    weak = (0.1 <= arr_label) & (arr_label < 1.0)
    mod = (1.0 <= arr_label) & (arr_label < 10.0)
    strong = (10 <= arr_label)
    if bin == "weak":
        return arr_pred[weak], arr_label[weak]
    if bin == "moderate":
        return arr_pred[mod], arr_label[mod]
    if bin == "strong":
        return arr_pred[strong], arr_label[strong]
    if bin == "no_rain":
        return arr_pred[no_rain], arr_label[no_rain]


def compute_index_bin(pred, label, out_type='df'):
    # All
    index_eval_all = compute_index(pred, label)
    # No-rain
    A, B = binning(pred, label, bin="no_rain")
    index_eval_no_rain = compute_index(A, B)
    # Weak rain
    A, B = binning(pred, label, bin="weak")
    index_eval_weak = compute_index(A, B)
    # Moderate rain
    A, B = binning(pred, label, bin="moderate")
    index_eval_moderate = compute_index(A, B)
    # Strong rain
    A, B = binning(pred, label, bin="strong")
    index_eval_strong = compute_index(A, B)
    # Result chart
    if out_type == 'df':
        out = pd.DataFrame([index_eval_all, index_eval_no_rain, index_eval_weak, index_eval_moderate, index_eval_strong], 
                    index = ['All', 'No_rain','Weak', 'Moderate', 'Strong'],
                    columns = ["n_pixel", "MAE", "MSE", "RMSE", "CC"]).T
    if out_type == 'arr':
        out = np.array([index_eval_all, index_eval_no_rain, index_eval_weak, index_eval_moderate, index_eval_strong])
    return out


# Conversion function in post-processing
def liner_transformation_polynomial(preds):
    # polynomial liner transformation
    X = preds.reshape(-1, 1)
    X = np.concatenate([X**2, X], axis=1)

    # y = ax**2 B bx + c
    a_b = np.array([0.01265849, 1.0173684])
    c = np.array([0.010581583])
    trans = np.matmul(X, a_b) + c
    trans = trans.reshape(preds.shape)
    return trans


# Tiles tp Image
def save_prediction(target_date = "2013080101", model_dir = "output_100epoch/PERSIAN"):
    # Load and recons data
    pred_image = get_image(dtype_index=0, target_dir=f"{model_dir}/Evaluation/dataset_112_full/test/{target_date}")
    # Save
    save_path = f"../prediction/{model_dir}"
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + "/" + target_date + ".npy", pred_image)

# datelist = pd.date_range(start="2013-08-01 00:00", end="2013-08-31 23:00", freq="H")
# datelist = [datelist[i].strftime('%Y%m%d%H') for i in range(len(datelist))]

def get_image(dtype_index=0, target_dir = "../data/dataset_64_full/test/2013080100"):
    """
    dtype_index = {0: precp, 1-2: wv, 3-4: ir, 5-6: cw, 7-8:ci}
    """
    patch = []
    for i in range(1, 17):# Iter 16(4*4) patch for a image
        arr = np.load(f"{target_dir}/{i}.npy")[dtype_index]# Get Precp.
        patch.append(arr)
    patch = np.array(patch)
    image = reconst_patch(patch, original_image_shape = (448, 448), window_shape = (112, 112))
    return image


def mask_norain(rainrate):
    return ma.masked_where(rainrate==0, rainrate)


def compute_index_with_mask(label, product, mask=True):
    if mask is True:
        product_ = ma.masked_where(product < -1, product).compressed()
        label_ = ma.masked_where(product < -1, label).compressed()
    elif mask is False:
        product_ = product
        label_ = label
    print("#evaluation data: ", len(product_), len(label_))
    evaluater = EvaluationIndices(product_, label_, rain_th_clsf=0.1, rain_th_reg=0)
    return evaluater.evaluate()


def compute_index_bin_with_mask(label, product, mask=True):
    if mask is True:
        product_ = ma.masked_where(product < -1, product).compressed()
        label_ = ma.masked_where(product < -1, label).compressed()
    elif mask is False:
        product_ = product
        label_ = label
    return compute_index_bin(product_, label_, out_type='df')


def compute_resid_with_mask(label, product, mask=True):
    if mask is True:
        product_ = ma.masked_where(product < -1, product).compressed()
        label_ = ma.masked_where(product < -1, label).compressed()
    elif mask is False:
        product_ = product
        label_ = label
    return product_ - label_