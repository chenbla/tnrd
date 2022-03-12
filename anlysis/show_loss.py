import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from anlysis.utils import get_loss_vec, smooth

output_dir_greedy = "C:/project/denoising_my/results/results_automation/output_dir_200_samples_original_sigma=50_epocs=2000/2022-03-07_12-32-49/"#"results/2022-03-04_19-18-26_-_3000_epocs/"
# output_dir_without_greedy = "results/2022-03-05_10-18-27_-_1000epocs_lr_0_no_reaction_term/"
# output_dir_l2_regularization_no_greedy = "results/2022-03-04_19-18-26_-_3000_epocs/"
# output_dir_DCT_regularization = "results/2022-03-04_19-18-26_-_3000_epocs/"

# def smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
#
# def get_last_losses_file(output_dir):
#     last_loss_file_path = None
#     index = 0
#     while True:
#         index += 10
#         loss_file_path = output_dir + "losses_e" + str(index) + ".pt"
#         if not os.path.isfile(loss_file_path):
#             break
#         last_loss_file_path = loss_file_path
#     return os.path.abspath(last_loss_file_path)
#
# def get_loss_vec(output_dir, loss_type='psnr'):
#     losses_file = get_last_losses_file(output_dir)
#     # print("losses file path: ", losses_file)
#     losses = torch.load(losses_file)
#     return losses[loss_type]



# Convergences

loss_vec = get_loss_vec(output_dir_greedy)
x = [e * 10 for e in list(range(1,len(loss_vec)+1))]
plt.plot(x, loss_vec, label="using greedy training - reference")
# plt.plot(get_loss_vec(output_dir_greedy), label="using skip connections")
# plt.plot(get_loss_vec(output_dir_greedy), label="using another activation method") #xUnit?

# Overfitting
# plt.plot(get_loss_vec(output_dir_greedy), label="using greedy training - reference")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="without using greedy")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="using l2 regularization")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="using kernel drop out")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="High frequency penalization")

plt.legend(loc="lower right")
plt.title("Training process optimization - avoid overfitting")
plt.show()


#plt.plot(smooth(np.array(losses_tnrd['G_recon']),50)[50:-50],'r')
#plt.plot(smooth(np.array(losses['G_recon']),50)[50:-50],'g')
#plt.legend(['Regularized with TNRD loss','Without TNRD loss'])
