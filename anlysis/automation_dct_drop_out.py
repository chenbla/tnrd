from subprocess import Popen, PIPE, STDOUT
import os

def run_command(command, workdir):
    print(command)
    p = Popen(command, stdout = PIPE, stderr = STDOUT, shell = True, cwd=workdir)
    for line in p.stdout:
        print(line.decode("utf-8"))#.replace('\n', ''))

def run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training, high_frequency_energy_weight, use_dct_drop_out, dont_use_dct, dct_weight_decay):
    arguments = "--root my_images --g-model g_tnrd --d-model d_tnrd --model-config \"{'gen_blocks':3, 'dis_blocks':4, 'in_channels':1}\" --reconstruction-weight 1.0 --perceptual-weight 0 --adversarial-weight 0 --tnrd-energy-weight 0 --crop-size 100 --gray-scale --noise-sigma " + str(sigma) + " --epochs " + str(
        epocs) + " --step-size 150 --batch-size 2 --eval-every 50 --print-every 100 --lr " + str(lr) + " --device cuda --use-tb --results-dir " + output_dir + " --train_max_size " + str(train_max_size) + " --high_frequency_energy_weight " + str(high_frequency_energy_weight)
    if use_greedy_training:
        arguments + " --use_greedy_training "
    if use_dct_drop_out:
        arguments + " --use_dct_drop_out "
    if dont_use_dct:
        arguments + " --dont_use_dct "
    if dct_weight_decay:
        arguments + " --dct_weight_decay "
    command = "python main.py " + arguments + " --dont_use_augmentation "
    run_command(command, code_folder)

sigma_sizes = [25]
train_sizes = [25, 50, 75, 125, 200, 300, 400]
epocs = 10
lr = 1e-3
high_frequency_energy_weight = 0
use_greedy_training = True
use_dct_drop_out = False
dont_use_dct = False
dct_weight_decay = False

for sigma in sigma_sizes:
    # # original
    # for train_size in train_sizes:
    #    train_max_size = train_size
    #    label = "original_sigma={}_epocs={}".format(sigma, epocs)
    #    output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
    #    code_folder = os.path.dirname(os.path.dirname(__file__)) #"C:/project/denoising_my/"
    #    run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training, high_frequency_energy_weight, use_dct_drop_out, dont_use_dct, dct_weight_decay)


    # # train with high frequency energy loss
    # high_frequency_energy_weight = 1
    # for train_size in train_sizes:
    #    train_max_size = train_size
    #    label = "high-frequency-loss_sigma={}_epocs={}_high_frequency_energy_weight={}".format(sigma, epocs, high_frequency_energy_weight)
    #    output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
    #    code_folder = os.path.dirname(os.path.dirname(__file__))
    #    run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training, high_frequency_energy_weight, use_dct_drop_out, dont_use_dct, dct_weight_decay)
    # high_frequency_energy_weight = 0

    # # without greedy training
    # use_greedy_training = False
    # for train_size in train_sizes:
    #    train_max_size = train_size
    #    label = "greedy-training_sigma={}_epocs={}".format(sigma, epocs)
    #    output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
    #    code_folder = os.path.dirname(os.path.dirname(__file__))
    #    run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training,
    #                  high_frequency_energy_weight, use_dct_drop_out, dont_use_dct, dct_weight_decay)
    # use_greedy_training = True

    # dct drop out
    use_dct_drop_out = True
    for train_size in train_sizes:
       train_max_size = train_size
       label = "dct-drop-out_sigma={}_epocs={}_dct_drop_out".format(sigma, epocs)
       output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
       code_folder = os.path.dirname(os.path.dirname(__file__))
       run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training,
                     high_frequency_energy_weight, use_dct_drop_out, dont_use_dct, dct_weight_decay)
    use_dct_drop_out = False

    # dont_use_dct
    # for train_size in train_sizes:
    #     dont_use_dct = True
    #
    #     train_max_size = train_size
    #     label = "dont-use-dct_sigma={}_epocs={}_dct_drop_out".format(sigma, epocs)
    #     output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
    #     code_folder = os.path.dirname(os.path.dirname(__file__))
    #     run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training,
    #                   high_frequency_energy_weight, use_dct_drop_out, dont_use_dct, dct_weight_decay)
    #
    # # dct_weight_decay
    # for train_size in train_sizes:
    #     dct_weight_decay = True
    #
    #     train_max_size = train_size
    #     label = "dct-weight-decay_sigma={}_epocs={}_dct_drop_out".format(sigma, epocs)
    #     output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
    #     code_folder = os.path.dirname(os.path.dirname(__file__))
    #     run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training,
    #                   high_frequency_energy_weight, use_dct_drop_out, dont_use_dct, dct_weight_decay)

    # optimize ssim


