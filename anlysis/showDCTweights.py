import torch
import matplotlib.pyplot as plt

weights_path = "C:/project/denoising_my/results/2022-03-05_14-54-37_-_500epocs_lr=1e-3/g_tnrd_e500.pt"
# weights_path = "C:/project/denoising_my/results/2022-03-05_14-54-37_-_500epocs_lr=1e-3/g_tnrd_e10.pt"
# weights_path = "C:/project/denoising_my/results/results_automation/output_dir_200_samples_original_sigma=50_epocs=2000/2022-03-07_12-32-49/g_tnrd_e2000.pt"
weights_path = "C:/project/denoising_my/results/results_automation__high_freq_regularization_partial/output_dir_75_samples_hish-frequency-loss_sigma=50_epocs=500_high_frequency_energy_weight=1/2022-03-12_12-21-46/g_tnrd_e500.pt"


#w1=torch.load('/home/itayh/Tec/xUnit/denoising/weight_rand_init.pt')
#wr=torch.load("/home/itayh/Tec/xUnit/denoising/weight2_rand_init.pt")
w1=torch.load(weights_path)
# wr=torch.load("/home/itayh/Tec/xUnit/denoising/weight2_init_rot.pt")

import numpy as np




# act_weights = w1['features_to_image.weight'][-2].cpu().detach().numpy()
# # act_weights.sum(axis=0).reshape(5,5)
# act_weights =np.concatenate((np.zeros(1), act_weights),0).reshape((5,5))
#
# print(act_weights)
# plt.imshow(act_weights)
# plt.show()


from matplotlib import gridspec

fig=plt.figure(figsize=(20,12))
gs = gridspec.GridSpec(5, 5)
for ind1 in range(5):
    for ind2 in range(5):
        ax = fig.add_subplot(gs[ind1, ind2])
        act_weights = w1['features_to_image.weight'][ind1*5+ind2-1].cpu().detach().numpy()
        FILTER = np.concatenate((np.zeros(1), act_weights), 0).reshape((5, 5))
        ax.imshow(FILTER)
plt.show()

# _, axs = plt.subplots(4, 6, figsize=(12, 12))
# axs = axs.flatten()
# for img, ax in zip(imgs, axs):
#     ax.imshow(img)
# plt.show()



# p=np.zeros((5,5))
# for ind1 in range(5):
#     for ind2 in range(5):
#         p[ind1,ind2]=ind1**2+ind2**2
# p=p/np.linalg.norm(p)
# plt.figure(); plt.imshow(p); plt.show()
