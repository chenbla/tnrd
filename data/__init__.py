import os
from .datasets import  DatasetNoise
from torch.utils.data import DataLoader

def get_loaders(args):
    # datasets
    # chen - before changes
    #dataset_train = DatasetNoise(root=os.path.join(args.root, 'train'), noise_sigma=args.noise_sigma, training=True, crop_size=args.crop_size, blind_denoising=args.blind, gray_scale=args.gray_scale)
    # chen - after changes
    dataset_train = DatasetNoise(root=os.path.join(args.root, 'train'), noise_sigma=args.noise_sigma, training=True, crop_size=args.crop_size, blind_denoising=args.blind, gray_scale=args.gray_scale, max_size=args.train_max_size, dont_use_augmentation=args.dont_use_augmentation)
    # dataset_train = DatasetNoise(root=os.path.join(args.root, 'train'), noise_sigma=args.noise_sigma, training=True, crop_size=args.crop_size, blind_denoising=args.blind, gray_scale=args.gray_scale, max_size=args.train_max_size)
    dataset_val = DatasetNoise(root=os.path.join(args.root, 'val'), noise_sigma=args.noise_sigma, training=False, crop_size=args.crop_size, blind_denoising=args.blind, gray_scale=args.gray_scale, max_size=args.max_size)
    dataset_train_eval = DatasetNoise(root=os.path.join(args.root, 'train'), noise_sigma=args.noise_sigma, training=False, crop_size=args.crop_size, blind_denoising=args.blind, gray_scale=args.gray_scale, max_size=args.max_size)

    # loaders
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader_eval = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)

    loader_train_for_eval = DataLoader(dataset_train_eval, batch_size=1, shuffle=False, num_workers=1)
    loaders = {'train': loader_train, 'eval': loader_eval, 'train_eval': loader_train_for_eval}

    return loaders
