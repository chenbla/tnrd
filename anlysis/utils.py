import torch
import numpy as np
import os
import math

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_last_losses_file(output_dir):
    last_loss_file_path = None
    index = 0
    while True:
        index += 10
        loss_file_path = output_dir + "losses_e" + str(index) + ".pt"
        if not os.path.isfile(loss_file_path):
            break
        last_loss_file_path = loss_file_path
    if last_loss_file_path is None:
        return None
    return os.path.abspath(last_loss_file_path)

def get_loss_vec(output_dir, loss_type='psnr'):
    losses_file = get_last_losses_file(output_dir)
    # print("losses file path: ", losses_file)
    if losses_file is None:
        return None
    losses = torch.load(losses_file)
    return losses[loss_type]

def get_train_max_psnr(output_dir):
    
    mse_loss_vec = get_loss_vec(output_dir, loss_type='psnr_train')
    min_mse = min(mse_loss_vec)
    return min_mse
