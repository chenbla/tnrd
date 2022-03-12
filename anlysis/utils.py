import torch
import numpy as np
import os


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
    return os.path.abspath(last_loss_file_path)

def get_loss_vec(output_dir, loss_type='psnr'):
    losses_file = get_last_losses_file(output_dir)
    # print("losses file path: ", losses_file)
    losses = torch.load(losses_file)
    return losses[loss_type]