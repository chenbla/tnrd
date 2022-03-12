import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt


def calc_loss(weights_from_torch, normalize=True):
    p = np.zeros((5, 5))
    for ind1 in range(5):
        for ind2 in range(5):
            p[ind1, ind2] = ind1 ** 2 + ind2 ** 2
    p = p / np.linalg.norm(p)
    # plt.figure()
    # plt.imshow(p)
    # plt.show()

    weights_matrix = calc_weights_sum(weights_from_torch, normalize=normalize)
    return np.multiply(weights_matrix, p).sum()


def show_all_weights(weights_from_torch):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(5, 5)
    for ind1 in range(5):
        for ind2 in range(5):
            ax = fig.add_subplot(gs[ind1, ind2])
            weights_row = weights_from_torch['features_to_image.weight'][ind1 * 5 + ind2 - 1].cpu().detach().numpy()
            weights_matrix = np.concatenate((np.zeros(1), weights_row), 0).reshape((5, 5))
            ax.imshow(weights_matrix)
    plt.show()


def calc_weights_sum(weights_from_torch, normalize = False):
    sum = np.zeros((5, 5))
    for ind1 in range(5):
        for ind2 in range(5):
            weights_row = weights_from_torch['features_to_image.weight'][ind1 * 5 + ind2 - 1].cpu().detach().numpy()
            weights_row += weights_from_torch['image_to_features.weight'][ind1 * 5 + ind2 - 1].cpu().detach().numpy()
            for i in [0, 1, 2]:
                weights_row += weights_from_torch["features.{}.weight".format(i)][ind1 * 5 + ind2 - 1].cpu().detach().numpy()
            weights_matrix = np.concatenate((np.zeros(1), weights_row), 0).reshape((5, 5))
            sum += weights_matrix

    if normalize:
        return sum/np.sum(sum.flatten())
    else:
        return sum


def show_avg_weights(weights_from_torch):
    weights_matrix = calc_weights_sum(weights_from_torch, normalize=True)

    fig, ax = plt.subplots()
    ax.matshow(weights_matrix, cmap=plt.cm.Blues)
    for i in range(weights_matrix.shape[0]):
        for j in range(weights_matrix.shape[1]):
            c = weights_matrix[j, i]
            ax.text(i, j, "{:.2f}".format(c), va='center', ha='center')
    plt.show()


def cmp_avg_weights(weights_from_torch_with_reg, weights_from_torch_without_reg):
    weights_matrix_with_reg = calc_weights_sum(weights_from_torch_with_reg, normalize=True)
    weights_matrix_without_reg = calc_weights_sum(weights_from_torch_without_reg, normalize=True)

    diff = weights_matrix_with_reg - weights_matrix_without_reg

    loss_without_reg = calc_loss(weights_from_torch_without_reg, normalize=True)
    loss_with_reg = calc_loss(weights_from_torch_with_reg, normalize=True)

    fig, axs = plt.subplots(1, 3)
    axs[0].matshow(weights_matrix_with_reg, cmap=plt.cm.Blues)
    axs[0].axis('off')
    axs[0].set_title('with regularization \n frequency loss ={:.2f}'.format(loss_with_reg))
    for i in range(weights_matrix_with_reg.shape[0]):
        for j in range(weights_matrix_with_reg.shape[1]):
            c = weights_matrix_with_reg[j, i]
            axs[0].text(i, j, "{:.2f}".format(c), va='center', ha='center')

    axs[1].matshow(weights_matrix_without_reg, cmap=plt.cm.Blues)
    axs[1].axis('off')
    axs[1].set_title('without regularization \n frequency loss ={:.2f}'.format(loss_without_reg))
    for i in range(weights_matrix_without_reg.shape[0]):
        for j in range(weights_matrix_without_reg.shape[1]):
            c = weights_matrix_without_reg[j, i]
            axs[1].text(i, j, "{:.2f}".format(c), va='center', ha='center')

    axs[2].matshow(diff, cmap=plt.cm.Blues)
    axs[2].axis('off')
    axs[2].set_title('with regularization - without regularization')
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            c = diff[j, i]
            axs[2].text(i, j, "{:.2f}".format(c), va='center', ha='center')

    plt.show()


weights_with_regularization_path = "C:/project/denoising_my/results/results_automation__high_freq_regularization_partial/output_dir_75_samples_hish-frequency-loss_sigma=50_epocs=500_high_frequency_energy_weight=1/2022-03-12_12-21-46/g_tnrd_e500.pt"
weights_without_regularization_path = "C:/project/denoising_my/results/results_automation__high_freq_regularization_partial/output_dir_75_samples_original_sigma=50_epocs=500/2022-03-12_01-50-59/g_tnrd_e500.pt"

# weights_with_regularization_path = "C:/project/denoising_my/results/2022-03-12_19-01-56_agressive_high_freq_loss/g_tnrd_e30.pt"

weights_without_reg = torch.load(weights_without_regularization_path)
weights_with_reg = torch.load(weights_with_regularization_path)

# show_all_weights(weights_with_reg)
# show_avg_weights(weights_with_reg)
cmp_avg_weights(weights_with_reg, weights_without_reg)

# loss_without_reg = calc_loss(weights_without_reg, normalize=True)
# loss_with_reg = calc_loss(weights_with_reg, normalize=True)
#
# print("loss with reg: {}, loss without reg: {} ".format(loss_with_reg, loss_without_reg))
