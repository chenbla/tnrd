import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from anlysis.utils import get_loss_vec, smooth
from statistics import mean

outputs_dir_main_folder = "C:/project/denoising_my/results/results_automation_overfitting/"#"results/overfitting_results/"
# outputs_dir_main_folder = "C:/project/denoising_my/results/results_automation_more_epocs/"#"results/overfitting_results/"

outputs_dir_main_folder = "C:/project/denoising_my/results/results_automation__high_freq_regularization_partial/"

color=['red','green','blue']

# output dir includes folder with name format "output_dir_400_samples_<label>"
output_dirs_list = glob.glob(outputs_dir_main_folder+"*/")
# print(output_dirs_list)

# get all the results fom specific label
output_dirs_dict = {}
for output_dir in output_dirs_list:
    label = (output_dir.split("_samples_",1)[1]).split("_",1)[0]
    # print(label)
    if label not in output_dirs_dict.keys():
        output_dirs_dict[label] = [output_dir]
    else:
        output_dirs_dict[label].append(output_dir)

labels = []
color_index = 1
for curr_label in output_dirs_dict.keys():
    curr_label_output_dir = output_dirs_dict[curr_label]
    for output_dir in curr_label_output_dir:
        output_dir = glob.glob(output_dir+"*/")[0]
        num_of_samples = int((output_dir.split("_samples_",1)[0]).split("_dir_",1)[1])
        # print(output_dir)
        # print(num_of_samples)
        # print(curr_label)

        loss_vec = get_loss_vec(output_dir)
        psnr = max(loss_vec)

        print(curr_label + str(psnr))

        if curr_label not in labels:
            labels.append(curr_label)
            plt.scatter(num_of_samples ,psnr, label=curr_label, color=color[color_index])
        else:
            plt.scatter(num_of_samples, psnr, color=color[color_index])
    color_index += 1

plt.legend(loc="lower right")
#plt.legend(labels)
plt.xlabel('Number of training samples')
plt.ylabel('Test PSNR(dB)')
plt.title("Training process optimization - avoid overfitting")
plt.show()

# output_dir_25_samples = "" #the training data size is 25
# output_dir_50_samples = ""
# output_dir_75_samples = ""
# output_dir_100_samples = ""
# output_dir_150_samples = ""
# output_dir_250_samples = ""
# output_dir_250_samples = ""
# output_dir_400_samples = ""

#
# # Overfitting
# plt.plot(get_loss_vec(output_dir_greedy), label="using greedy training - reference")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="without using greedy")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="using l2 regularization")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="using kernel drop out")
# plt.plot(get_loss_vec(output_dir_without_greedy), label="High frequency penalization")
#
# plt.legend(loc="lower right")
# plt.title("Training process optimization - avoid overfitting")
# plt.show()
