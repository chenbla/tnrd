import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from anlysis.utils import get_loss_vec, smooth

output_dir_greedy = "C:/project/denoising_my/results/results_automation/output_dir_200_samples_original_sigma=50_epocs=2000/2022-03-07_12-32-49/"#"results/2022-03-04_19-18-26_-_3000_epocs/"
# output_dir_without_greedy = "results/2022-03-05_10-18-27_-_1000epocs_lr_0_no_reaction_term/"
# output_dir_l2_regularization_no_greedy = "results/2022-03-04_19-18-26_-_3000_epocs/"
# output_dir_DCT_regularization = "results/2022-03-04_19-18-26_-_3000_epocs/"

output_dir_greedy = "../results/results_automation/output_dir_400_samples_high-frequency-loss_sigma=25_epocs=700_high_frequency_energy_weight=1/2022-03-22_22-03-47/"


loss_vec = get_loss_vec(output_dir_greedy)
x = [e * 10 for e in list(range(1,len(loss_vec)+1))]
plt.plot(x, loss_vec, label="using greedy training - reference")
plt.legend(loc="lower right")
plt.title("Training process optimization - avoid overfitting")
plt.show()

#plt.plot(smooth(np.array(losses_tnrd['G_recon']),50)[50:-50],'r')
