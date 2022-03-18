import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import shutil

epocs = 700
outputs_dir_main_folder = "../results/results_automation/"

outputs_dir_minimal = os.path.join("../results/results_automation_minimal/")
print(outputs_dir_minimal)
os.makedirs(outputs_dir_minimal, exist_ok=True)

output_dirs_list=[]
# output dir includes folder with name format "output_dir_400_samples_<label>"
outputs_dir_main_folder_full_path = os.path.abspath(outputs_dir_main_folder)
print(outputs_dir_main_folder_full_path)
output_dirs_names_list = os.listdir(outputs_dir_main_folder_full_path)
for dir_name in output_dirs_names_list:
    output_dirs_list.append(os.path.join(outputs_dir_main_folder_full_path, dir_name))
# print(output_dirs_list)


# get all the results fom specific label
output_dirs_dict = {}
for output_dir in output_dirs_list:
    # print(output_dir)
    label = (output_dir.split("_samples_",1)[1]).split("_",1)[0]
    # print(label)
    if label not in output_dirs_dict.keys():
        output_dirs_dict[label] = [output_dir]
    else:
        output_dirs_dict[label].append(output_dir)

# print(output_dirs_dict)


labels = []
color_index = 1
for curr_label in output_dirs_dict.keys():
    curr_label_output_dir = output_dirs_dict[curr_label]

    curr_label_minimal_dir = os.path.join(outputs_dir_minimal, curr_label)
    # os.makedirs(curr_label_minimal_dir, exist_ok=True)

    for output_dir in curr_label_output_dir:
        output_dir = glob.glob(output_dir+"*/")[0]
        num_of_samples = int((output_dir.split("_samples_",1)[0]).split("_dir_",1)[1])
        # print(output_dir)
        # print(num_of_samples)
        # print(curr_label)

        minimal_path = os.path.join(curr_label_minimal_dir, os.path.basename(output_dir[:-1]))
        # print(minimal_path)
        output_dirs_names_list = os.listdir(output_dir)
        # print(output_dirs_names_list)
        for date in output_dirs_names_list:
            output_dir =  os.path.join(output_dir , date)
            output_dir_minimal = os.path.join(minimal_path, date)
            print(output_dir)
            # print(output_dir_minimal)

            os.makedirs(output_dir_minimal, exist_ok=True)

            from os import listdir
            from os.path import isfile, join
            files_in_output_dir = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]
            for file in files_in_output_dir:
                for i in range(100,epocs+1, 100):
                    if str(i) in file:
                        print(file)

                        src = os.path.join(output_dir, file)
                        dst = os.path.join(output_dir_minimal, file)
                        if not os.path.exists(dst):
                            shutil.copy(src, dst)

        shutil.copy(os.path.join(output_dir, "log.txt"), os.path.join(output_dir_minimal, "log.txt"))








