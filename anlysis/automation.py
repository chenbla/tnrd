from subprocess import Popen, PIPE, STDOUT
import os

def run_command(command, workdir):
    p = Popen(command, stdout = PIPE, stderr = STDOUT, shell = True, cwd=workdir)
    for line in p.stdout:
        print(line.decode("utf-8"))#.replace('\n', ''))

def run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training, high_frequency_energy_weight):
    arguments = "--root my_images --g-model g_tnrd --d-model d_tnrd --model-config \"{'gen_blocks':3, 'dis_blocks':4, 'in_channels':1}\" --reconstruction-weight 1.0 --perceptual-weight 0 --adversarial-weight 0 --tnrd-energy-weight 0 --crop-size 100 --gray-scale --noise-sigma " + str(sigma) + " --epochs " + str(
        epocs) + " --step-size 150 --batch-size 2 --eval-every 10 --print-every 100 --lr " + str(lr) + " --device cuda --use-tb --results-dir " + output_dir + " --train_max_size " + str(train_max_size) + " --high_frequency_energy_weight " + str(high_frequency_energy_weight)
    if use_greedy_training:
        arguments + " --use_greedy_training "
    command = "python main.py " + arguments
    run_command(command, code_folder)


train_sizes = [25, 50, 75, 125, 200, 300, 400]

# train without greedy training
for train_size in train_sizes:
    high_frequency_energy_weight = 0
    use_greedy_training = False
    sigma = 50
    epocs = 500
    lr = 1e-3
    train_max_size = train_size
    label = "original_sigma={}_epocs={}".format(sigma, epocs)
    output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
    code_folder = os.path.dirname(os.path.dirname(__file__)) #"C:/project/denoising_my/"
    run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training, high_frequency_energy_weight)
#
# # train with greedy training
# for train_size in train_sizes:
#     use_greedy_training = True
#     sigma = 50
#     epocs = 2000
#     lr = 1e-3
#     train_max_size = train_size
#     label = "original-greedy_sigma={}_epocs={}".format(sigma, epocs)
#     output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
#     code_folder = os.path.dirname(os.path.dirname(__file__)) #"C:/project/denoising_my/"
#     run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training)


# train with high frequency energy loss
for train_size in train_sizes:
    high_frequency_energy_weight=1
    use_greedy_training = False
    sigma = 50
    epocs = 500
    lr = 1e-3
    train_max_size = train_size
    label = "hish-frequency-loss_sigma={}_epocs={}_high_frequency_energy_weight={}".format(sigma, epocs, high_frequency_energy_weight)
    output_dir = "./results/results_automation/output_dir_{}_samples_{}".format(train_max_size, label)
    code_folder = os.path.dirname(os.path.dirname(__file__))
    run_denoising(code_folder, sigma, epocs, lr, output_dir, train_max_size, use_greedy_training, high_frequency_energy_weight)
