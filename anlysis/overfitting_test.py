import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from anlysis.utils import get_loss_vec, get_train_max_psnr, smooth
from statistics import mean

outputs_dir_main_folder = "C:/project/denoising_my/results/results_automation_overfitting/"

# outputs_dir_main_folder = "../results/results_automation_before_new_dataset/"#"../results/results_automation/"#sigma=50
outputs_dir_main_folder = "../results/results_automation/"#sigma=50
# outputs_dir_main_folder = "/home/chenkatz@staff.technion.ac.il/tnrd/results/results_automation/" #sigma=25

# outputs_dir_main_folder = "/home/chenkatz/tnrd/results/results_automation/" #sigma=25


color=['red','green','blue', 'black', 'brown']



def cmp_test_psnr():
    # get all the results fom specific label
    output_dirs_dict = {}
    for output_dir in output_dirs_list:
        label = (output_dir.split("_samples_", 1)[1]).split("_", 1)[0]
        # print(label)
        if label not in output_dirs_dict.keys():
            output_dirs_dict[label] = [output_dir]
        else:
            output_dirs_dict[label].append(output_dir)

    labels = []
    color_index = 0
    for curr_label in output_dirs_dict.keys():
        curr_label_output_dir = output_dirs_dict[curr_label]



        if "greedy" in curr_label_output_dir[0]:
            continue

        # if "high" in curr_label_output_dir[0]:
            # continue

        # didnt finished yet
        if "dct-weight-decay" in curr_label_output_dir[0]:
            continue

        if "dont-use-dct" in curr_label_output_dir[0]:
            continue

        # print(curr_label_output_dir[0])

        for output_dir in curr_label_output_dir:
            output_dir = glob.glob(output_dir+"*/")[0]
            num_of_samples = int((output_dir.split("_samples_",1)[0]).split("_dir_",1)[1])
            print(output_dir)
            # print(num_of_samples)
            # print(curr_label)

            #get test loss vector
            loss_vec = get_loss_vec(output_dir)
            if loss_vec is None:
                print("cant find loss vec in: " + output_dir)
                continue
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


# output dir includes folder with name format "output_dir_400_samples_<label>"
output_dirs_list = glob.glob(outputs_dir_main_folder+"*/")
print(output_dirs_list)



def cmp_to_original(tag, show_train, show_test):
    # show train and test with and without regularozation
    # get all the results fom specific label
    output_dirs_dict = {}
    for output_dir in output_dirs_list:
        label = (output_dir.split("_samples_", 1)[1]).split("_", 1)[0]
        # print(label)
        if label not in output_dirs_dict.keys():
            output_dirs_dict[label] = [output_dir]
        else:
            output_dirs_dict[label].append(output_dir)

    labels = []
    color_index = 0
    for curr_label in output_dirs_dict.keys():
        curr_label_output_dir = output_dirs_dict[curr_label]



        if (tag not in curr_label_output_dir[0]) and ('original' not in curr_label_output_dir[0]):
            print(curr_label_output_dir[0])
            continue


        # print(curr_label_output_dir[0])

        for output_dir in curr_label_output_dir:
            output_dir = glob.glob(output_dir+"*/")[0]
            num_of_samples = int((output_dir.split("_samples_",1)[0]).split("_dir_",1)[1])
            print(output_dir)
            # print(num_of_samples)
            # print(curr_label)

            # get train loss
            train_psnr = get_train_max_psnr(output_dir)

            #get test loss vector
            loss_vec = get_loss_vec(output_dir)
            if loss_vec is None:
                print("cant find loss vec in: " + output_dir)
                continue
            test_psnr = max(loss_vec)
            # test_psnr = np.median(loss_vec)

            print(curr_label +": test: " + str(test_psnr) + ", train: " + str(train_psnr))


            if show_test:
                if curr_label+"test" not in labels:
                    labels.append(curr_label+"test")
                    plt.scatter(num_of_samples ,test_psnr, label=curr_label+"test", color=color[color_index])
                else:
                    plt.scatter(num_of_samples, test_psnr, color=color[color_index])

            if show_train and train_psnr is not None:
                if curr_label+"train" not in labels:
                    labels.append(curr_label+"train")
                    plt.scatter(num_of_samples ,train_psnr, label=curr_label+"train", color=color[color_index+1])
                else:
                    plt.scatter(num_of_samples, train_psnr, color=color[color_index+1])

        color_index += 2

    plt.legend(loc="lower right")
    #plt.legend(labels)
    plt.xlabel('Number of training samples')
    plt.ylabel('Test PSNR(dB)')
    plt.title("Training process optimization - avoid overfitting")
    plt.show()


# cmp_test_psnr()
cmp_to_original(tag = "high-frequency-loss", show_train=True, show_test=True)

# cmp_to_original(tag = "dct-weight-decay", show_train=True, show_test=True)
