images_dir = "../my_images"
train_original_path = images_dir + "/train"
val_original_path = images_dir + "/val"
new_train_path = images_dir + "/train_new"
new_val_path = images_dir + "/val_new"

import os
from os import listdir
from os.path import isfile, join
import random
import shutil

# get the list of all the images:
all_test_images_paths = [os.path.join(train_original_path, f) for f in listdir(train_original_path) if isfile(join(train_original_path, f))]
all_val_images_paths = [os.path.join(val_original_path, f) for f in listdir(val_original_path) if isfile(join(val_original_path, f))]
all_the_images = all_test_images_paths + all_val_images_paths


# split a dataset into train and test sets
random.shuffle(all_the_images)
new_train_paths = all_the_images[len(all_val_images_paths):]
vew_val_paths = all_the_images[:len(all_val_images_paths)]

print(len(new_train_paths), len(new_val_path))

# create new_folders:
if os.path.isdir(new_train_path) or os.path.isdir(new_val_path):
    print("new folders already exist")
    exit()

os.makedirs(new_train_path)
os.makedirs(new_val_path)

for source in new_train_paths:
    print(source)
    dest = shutil.copy(source, new_train_path)
    print(dest)
    print("--")

print("================================================================")

for source in vew_val_paths:
    print(source)
    dest = shutil.copy(source, new_val_path)
    print(dest)
    print("--")


