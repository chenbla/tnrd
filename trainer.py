import random
import logging
import models
import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad as torch_grad, Variable
from data import get_loaders
from ast import literal_eval
from utils.recorderx import RecoderX
from utils.misc import save_image, average, mkdir, compute_psnr
from models.modules.losses import RangeLoss, PerceptualLoss, TexturalLoss
from functools import partial
from models.tnrd import TNRDConv2d, TNRDlayer
from models.modules.activations import RBF
import torchvision
import copy
import math
from torch.autograd.function import InplaceFunction
from PIL import Image
import numpy as np

from ssim.pytorch_ssim import ssim

_EImage = -1


# class Round(InplaceFunction):
#
#     @staticmethod
#     def forward(ctx, input,inplace):
#
#         ctx.inplace = inplace
#         if ctx.inplace:
#             ctx.mark_dirty(input)
#             output = input
#         else:
#             output = input.clone()
#         output.round_()
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # straight-through estimator
#         grad_input = grad_output
#         return grad_input,None

# from piqa import SSIM #ERROR: torchaudio 0.8.1 has requirement torch==1.8.1, but you'll have torch 1.11.0 which is incompatible.
#
# class SSIMLoss(SSIM):
#     def forward(self, x, y):
#         return 1. - super().forward(x, y)

class L2Loss_image(torch.nn.Module):
    """docstring for L2Loss_image."""

    def __init__(self):
        super(L2Loss_image, self).__init__()
        # self.stochastic = False

    def forward(self, x, y):
        # if self.stochastic:
        #     noise = x.new(x.shape).uniform_(-0.5, 0.5)
        #     x.add_(noise)
        return l2_loss_image(x, y)


def l2_loss_image(x, y):
    x = x.clamp(0, 255)
    y = y.clamp(0, 255)
    return (x - y).pow(2).mean()


class Trainer():
    def __init__(self, args):
        # parameters
        self.args = args
        self.device = args.device
        self.session = 0
        self.print_model = True
        self.invalidity_margins = None

        if self.args.use_tb:
            self.tb = RecoderX(log_dir=args.save_path)

        # initialize
        self._init()

    def _init_model(self):
        # initialize model
        if self.args.model_config != '':
            model_config = dict({}, **literal_eval(self.args.model_config))
        else:
            model_config = {}

        model_config['all_args'] = self.args

        g_model = models.__dict__[self.args.g_model]
        self.g_model = g_model(**model_config)

        # loading weights
        if self.args.gen_to_load != '':
            logging.info('\nLoading g-model...')
            self.g_model.load_state_dict(torch.load(self.args.gen_to_load, map_location='cpu'))

        # to cuda
        self.g_model = self.g_model.to(self.args.device)

        # parallel
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.g_model = torch.nn.DataParallel(self.g_model, self.args.device_ids)

        # print model
        if self.print_model:
            logging.info(self.g_model)
            logging.info('Number of parameters in generator: {}\n'.format(
                sum([l.nelement() for l in self.g_model.parameters()])))
            self.print_model = False

    def _init_optim(self):
        # initialize optimizer
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=self.args.lr, betas=self.args.gen_betas,
                                            weight_decay=self.args.dct_weight_decay)
        # self.g_optimizer = torch.optim.SGD(self.g_model.parameters(), lr=self.args.lr,momentum=0.9)

        # initialize scheduler
        self.g_scheduler = StepLR(self.g_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # initialize criterion
        if self.args.reconstruction_weight:
            self.reconstruction = L2Loss_image().to(
                self.device)  # torch.nn.L1Loss().to(self.args.device) #torch.nn.MSELoss().to(self.args.device)
        if self.args.perceptual_weight > 0.:
            self.perceptual = PerceptualLoss(features_to_compute=['conv5_4'], criterion=torch.nn.L1Loss(),
                                             shave_edge=self.invalidity_margins).to(self.args.device)
        if self.args.textural_weight > 0.:
            self.textural = TexturalLoss(features_to_compute=['relu3_1', 'relu2_1'],
                                         shave_edge=self.invalidity_margins).to(self.args.device)
        if self.args.range_weight > 0.:
            self.range = RangeLoss(invalidity_margins=self.invalidity_margins).to(self.args.device)

    def _init(self):
        # init parameters
        self.steps = 0
        self.losses = {'D': [], 'D_r': [], 'D_gp': [], 'D_f': [], 'G': [], 'G_recon': [], 'G_rng': [], 'G_perc': [],
                       'G_txt': [], 'G_adv': [], 'psnr': [], 'best_model_psnr': [], 'tnrd_loss': [],
                       'high_freq_loss': [], 'psnr_train': [], 'ssim_train': [], 'ssim_test': []}

        # initialize model
        self._init_model()

        # initialize optimizer
        self._init_optim()

        self.cached_output = {}
        self.cached_input = {}
        self.pad_input = torchvision.transforms.Pad(5, padding_mode='edge')

        def hook(name, module, input, output):
            if isinstance(module, RBF):
                self.cached_output[name] = output[1]
            else:
                # self.cached_input[name] = input[0][0]
                self.cached_input[name] = output[0]

        self.handlers = []
        for name, m in self.g_model.named_modules():
            if (isinstance(m, RBF) or isinstance(m, TNRDConv2d) or isinstance(m, TNRDlayer)):
                self.handlers.append(m.register_forward_hook(partial(hook, name)))

    def tnrd_loss(self, f):
        loss = 0
        for key_u, key_r in zip(self.cached_input.keys(), self.cached_output.keys()):
            layer_loss = self.cached_output[key_r]  # +0.5*(self.cached_input[key_u]-self.pad_input(f)).pow(2)
            loss += layer_loss.mean()
        return torch.exp(loss / 20)

    def greedy_loss(self, target):
        loss = 0
        for ind, key in enumerate(self.cached_input.keys()):
            layer_loss = l2_loss_image(self.cached_input[key], target)  # F.l1_loss(self.cached_input[key],target)
            loss += layer_loss.mean()
        return loss

    def layer_loss(self, target, loss_from_layer):
        print("layer_loss--"+str(loss_from_layer))
        layer_input_ind = 1  # 1-5
        for ind, key in enumerate(self.cached_input.keys()):
            # for ind, key in enumerate(self.cached_output.keys()):
            if (layer_input_ind - 1) == loss_from_layer:
                l2_loss = l2_loss_image(self.cached_input[key], target)  # F.l1_loss(self.cached_input[key],target)
                ssim_loss = (1 - ssim(self.cached_input[key], target))
                return l2_loss, ssim_loss
            layer_input_ind += 1

    def _save_model(self, epoch):
        # save models
        torch.save(self.g_model.state_dict(),
                   os.path.join(self.args.save_path, '{}_e{}.pt'.format(self.args.g_model, epoch + 1)))
        # torch.save(self.d_model.state_dict(), os.path.join(self.args.save_path, '{}_e{}.pt'.format(self.args.d_model, epoch + 1)))
        torch.save(self.losses, os.path.join(self.args.save_path, 'losses_e{}.pt'.format(epoch + 1)))

    def _generator_iteration(self, inputs, targets):
        # zero grads
        self.g_optimizer.zero_grad()

        # get generated data
        generated_data = self.g_model(inputs)
        loss = 0.

        # # reconstruction loss
        # if self.args.reconstruction_weight > 0.:
        #     loss_recon = self.reconstruction(generated_data, targets)
        #     loss += loss_recon * self.args.reconstruction_weight
        #     self.losses['G_recon'].append(loss_recon.data.item())

        # # ssim reconstruction loss
        # if self.args.ssim_reconstruction_weight > 0.:
        #     # https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
        #     from ssim.pytorch_ssim import ssim
        #     ssim_out = 200*(1-ssim(generated_data, targets))#.data#.item()  # [0]
        #     loss += ssim_out

        # high freq regularization
        if self.args.high_frequency_energy_weight > 0.:
            dct_penalty_matrix = np.zeros((5, 5))
            for ind1 in range(5):
                for ind2 in range(5):
                    dct_penalty_matrix[ind1, ind2] = ind1 ** 2 + ind2 ** 2
            dct_penalty_matrix = dct_penalty_matrix / np.linalg.norm(dct_penalty_matrix)

            dct_penalty_matrix = torch.tensor(dct_penalty_matrix).to(self.device)
            final_dct_penalty_matrix = torch.diag(dct_penalty_matrix.flatten().float()[1:])

            # test:
            # t = torch.ones(24, 24).to(self.device)
            # res = t @ final_dct_penalty_matrix

            high_freq_loss = torch.tensor(0., requires_grad=True)
            for name, param in self.g_model.features_to_image.named_parameters():
                if 'weight' == name:
                    high_freq_loss = high_freq_loss + torch.norm(param @ final_dct_penalty_matrix)
            for name, param in self.g_model.image_to_features.named_parameters():
                if 'weight' == name:
                    high_freq_loss = high_freq_loss + torch.norm(param @ final_dct_penalty_matrix)
            for name, param in self.g_model.features.named_parameters():
                if ('weight' in name) and ('act' not in name):
                    high_freq_loss = high_freq_loss + torch.norm(param @ final_dct_penalty_matrix)

            self.losses['high_freq_loss'].append(high_freq_loss)

            # print("loss{}, high frequency loss: {}".format(loss, high_freq_loss))
            loss = loss + self.args.high_frequency_energy_weight * high_freq_loss
        ######################################

        # chen: greedy - 1/3 of the epocs
        # if self.args.use_greedy_training:
        #     if self.steps<0.3*self.args.epochs * self.args.train_max_size: #2e4:
        #        loss += self.greedy_loss(targets)


        #
        total_number_of_layers = dict({}, **literal_eval(self.args.model_config))['gen_blocks']+2  # 5
        number_of_epocs_of_each_layer = np.floor(0.5 * self.args.epochs / (total_number_of_layers - 1))
        if self.args.use_greedy_training:
            layer_to_train = int(np.floor(self.current_epoc / number_of_epocs_of_each_layer)) +1
            if number_of_epocs_of_each_layer == 1:
                layer_to_train -= 1
            if layer_to_train < total_number_of_layers:
                # layer_to_train+=1

                # print("steps={}, layer_to_train={}".format(self.steps, layer_to_train))

                for name, param in self.g_model.image_to_features.named_parameters():
                    if layer_to_train == 1:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                for name, param in self.g_model.features.named_parameters():
                    layer_ind = 1 + (int(name[0]) + 1)
                    if layer_ind == layer_to_train:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                for name, param in self.g_model.features_to_image.named_parameters():
                    if layer_ind == layer_to_train:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

                l2_loss, ssim_loss = self.layer_loss(targets, layer_to_train)
                loss += l2_loss * self.args.reconstruction_weight
                loss += ssim_loss * self.args.ssim_reconstruction_weight
                self.losses['G_recon'].append(l2_loss.data.item())

                # print(loss)

                ##
                # print requre grad status:
                # for name, param in self.g_model.image_to_features.named_parameters():
                #     print(param.requires_grad)

                ##

            else:
                if layer_to_train == total_number_of_layers:
                    for name, param in self.g_model.image_to_features.named_parameters():
                        param.requires_grad_(True)
                    for name, param in self.g_model.features.named_parameters():
                        param.requires_grad_(True)
                    for name, param in self.g_model.features_to_image.named_parameters():
                        param.requires_grad_(True)

                # reconstruction loss
                if self.args.reconstruction_weight > 0.:
                    loss_recon = self.reconstruction(generated_data, targets)
                    loss += loss_recon * self.args.reconstruction_weight
                    self.losses['G_recon'].append(loss_recon.data.item())

                if self.args.ssim_reconstruction_weight > 0.:
                    # https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
                    ssim_out = (1 - ssim(generated_data, targets))  # .data#.item()  # [0]
                    # ssim_value = ssim(generated_data, targets)
                    loss += ssim_out * self.args.ssim_reconstruction_weight
        else:
            # reconstruction loss
            if self.args.reconstruction_weight > 0.:
                loss_recon = self.reconstruction(generated_data, targets)
                loss += loss_recon * self.args.reconstruction_weight
                self.losses['G_recon'].append(loss_recon.data.item())

            # ssim reconstruction loss
            if self.args.ssim_reconstruction_weight > 0.:
                # https://stackoverflow.com/questions/53956932/use-pytorch-ssim-loss-function-in-my-model
                ssim_out = (1 - ssim(generated_data, targets))  # .data#.item()  # [0]
                # ssim_value = ssim(generated_data, targets)
                loss += ssim_out * self.args.ssim_reconstruction_weight
        ###



        # if len(self.losses['G_recon'])>1 and self.losses['G_recon'][-1]< min(self.losses['G_recon'][:-1]):
        #    print('new best G_recon model: ', self.losses['G_recon'][-1])
        #    torch.save(self.g_model.state_dict(),'best_g_model.pth')
        # backward loss
        loss.backward()
        # debug
        # for pp in self.g_model.parameters():
        #    if pp.grad is not None:
        #        print(pp.shape,pp.max(),pp.min(),pp.mean())
        #        print('grad - ',pp.grad.max(),pp.grad.min(),pp.grad.mean())
        #        import pdb; pdb.set_trace()
        self.g_optimizer.step()

        # record loss
        self.losses['G'].append(loss.data.item())

    def _train_iteration(self, data):
        # set inputs
        self.steps += 1
        inputs = data['input'].to(self.device)
        targets = data['target'].to(self.device)

        # chen: i removed it
        # # critic iteration
        # if self.args.adversarial_weight > 0.:
        #     if self.args.wgan:
        #         self._critic_wgan_iteration(inputs, targets)
        #     else:
        #         self._critic_hinge_iteration(inputs, targets)

        # only update generator every |critic_iterations| iterations
        if self.steps % self.args.num_critic == 0:
            self._generator_iteration(inputs, targets)

        # logging
        if self.steps % self.args.print_every == 0:
            line2print = 'Iteration {}'.format(self.steps + 1)
            if self.args.adversarial_weight > 0.:
                line2print += ', D: {:.6f}, D_r: {:.6f}, D_f: {:.6f}'.format(self.losses['D'][-1],
                                                                             self.losses['D_r'][-1],
                                                                             self.losses['D_f'][-1])
                if self.args.penalty_weight > 0.:
                    line2print += ', D_gp: {:.6f}'.format(self.losses['D_gp'][-1])
            if self.steps > self.args.num_critic:
                line2print += ', G: {:.5f}'.format(sum(self.losses['G'][-100:]) / 100)
                if self.args.reconstruction_weight:
                    line2print += ', G_recon: {:.6f}'.format(sum(self.losses['G_recon'][-100:]) / 100)
                if self.args.range_weight:
                    line2print += ', G_rng: {:.6f}'.format(self.losses['G_rng'][-1])
                if self.args.perceptual_weight:
                    line2print += ', G_perc: {:.6f}'.format(self.losses['G_perc'][-1])
                if self.args.textural_weight:
                    line2print += ', G_txt: {:.8f}'.format(self.losses['G_txt'][-1])
                if self.args.adversarial_weight:
                    line2print += ', G_adv: {:.6f},'.format(self.losses['G_adv'][-1])
                # if True:
                #    line2print += ', tnrd_loss: {:.6f},'.format(self.losses['tnrd_loss'][-1])

                if self.args.high_frequency_energy_weight:
                    line2print += ', high_freq_loss: {:.6f},'.format(self.losses['high_freq_loss'][-1])
            logging.info(line2print)

        # plots for tensorboard
        if self.args.use_tb:
            if self.args.adversarial_weight > 0.:
                self.tb.add_scalar('data/loss_d', self.losses['D'][-1], self.steps)
            if self.steps > self.args.num_critic:
                self.tb.add_scalar('data/loss_g', self.losses['G'][-1], self.steps)

    def _eval_iteration(self, data, epoch, ii):
        # set inputs
        inputs = data['input'].to(self.device)
        targets = data['target']
        paths = data['path']

        # evaluation

        # test:
        # import matplotlib.pyplot as plt
        # plt.imshow(inputs[0][0].numpy())
        # plt.show()

        with torch.no_grad():
            outputs = self.g_model(inputs)
            # if ii==_EImage:  # chen: i remove it!
        if False:  # chen: i added it
            image = Image.fromarray(inputs.squeeze().squeeze().clamp(0, 255).round().cpu().numpy().astype(np.uint8))
            image.save('noisy_%s.png' % (self.args.noise_sigma))
            image = Image.fromarray(outputs.squeeze().squeeze().clamp(0, 255).round().cpu().numpy().astype(np.uint8))
            image.save('clean_%s.png' % (self.args.noise_sigma))
            image = Image.fromarray(targets.squeeze().squeeze().clamp(0, 255).round().cpu().numpy().astype(np.uint8))
            image.save('target_%s.png' % (self.args.noise_sigma))
            # save image and compute psnr
        self._save_image(outputs, paths[0], epoch + 1)
        psnr = compute_psnr(outputs, targets)

        return psnr

    def _train_epoch(self, loader):
        self.g_model.train()

        # train over epochs
        for _, data in enumerate(loader):
            self._train_iteration(data)

    def _eval_epoch(self, loader, epoch, loader_train=None):
        # sd_model = copy.deepcopy(self.g_model.state_dict())
        self.g_model.eval()
        psnrs = []
        # eval over epoch
        for ii, data in enumerate(loader):
            psnr = self._eval_iteration(data, epoch, ii)
            psnrs.append(psnr)
        # record psnr
        self.losses['psnr'].append(average(psnrs))

        psnrs_train = []
        # ssim_train = []
        if loader_train is not None:
            for ii, data in enumerate(loader_train):
                psnr = self._eval_iteration(data, epoch, ii)
                psnrs_train.append(psnr)

                # ssim = (1 - ssim(generated_data, targets))
                # ssim_train.append()
            # record psnr
            self.losses['psnr_train'].append(average(psnrs_train))

        if loader_train is None:
            logging.info('Evaluation: {:.3f}'.format(self.losses['psnr'][-1]))
        else:
            logging.info("Evaluation: {:.3f},      Train-evaluation: {:.3f}".format(self.losses['psnr'][-1],
                                                                                    self.losses['psnr_train'][-1]))
        if self.args.use_tb:
            self.tb.add_scalar('data/psnr', self.losses['psnr'][-1], epoch)

    def _save_image(self, image, path, epoch):
        directory = os.path.join(self.args.save_path, 'images', 'epoch_{}'.format(epoch))
        save_path = os.path.join(directory, os.path.basename(path))
        mkdir(directory)
        save_image(image.data.cpu(), save_path)

    def _train(self, loaders):
        self.current_epoc = 0
        # run epoch iterations
        for epoch in range(self.args.epochs):
            self.current_epoc += 1

            # random seed
            torch.manual_seed(random.randint(1, 123456789))

            logging.info('\nEpoch {}'.format(epoch + 1))

            # train
            self._train_epoch(loaders['train'])

            # scheduler
            self.g_scheduler.step(epoch=epoch)
            # self.d_scheduler.step(epoch=epoch)

            # evaluation
            if ((epoch + 1) % self.args.eval_every == 0) or ((epoch + 1) == self.args.epochs):
                self._eval_epoch(loaders['eval'], epoch, loaders['train_eval'])
                self._save_model(epoch)

        # best score

        logging.info('Best PSNR Score: {:.2f}\n'.format(max(self.losses['psnr'])))

    def train(self):
        # get loader
        loaders = get_loaders(self.args)

        # run training
        self._train(loaders)

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()

    def eval(self):
        # get loader
        loaders = get_loaders(self.args)

        # evaluation
        logging.info('\nEvaluating...')
        self._eval_epoch(loaders['eval'], 0)

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()
