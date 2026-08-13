import os
import torch
from torch import nn
from MultiNet import eval_tools
import numpy as np
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from MultiNet.loss import crossview_contrastive_Loss, KLD_Loss
from sklearn.cluster import KMeans


class Trainer:
    def __init__(self, data, net, device):
        self.batch_size = None
        self.data = data
        self.network_ae = net.ae_network.to(device)
        self.network_d = net.d_network.to(device)
        self.view_num = 2
        self.device = device

        self.reconstruction_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        self.contrastive_loss = crossview_contrastive_Loss()
        self.mse_loss = nn.MSELoss()
        self.kld_loss = KLD_Loss()

    def train(self, configs):
        use_indicator = configs['use_indicator']
        configs = configs[f'{self.data.dataname}_Params']
        print("Pretrain...")
        self._pretrain_ae_d(configs['batch_size'],
                            configs["pretrain_epochs"],
                            configs['lr_ae'],
                            configs['lr_d'],
                            configs['alpha'],
                            configs['zeta_1'],
                            configs['zeta_2'],
                            configs['noise_sigma'],
                            use_indicator)
        print("==> Pretrain Done")
        print("Train...")
        self._train_ae_d(configs['batch_size'],
                        configs["train_epochs"],
                        configs['cutoff'],
                        configs['lr_ae'],
                        configs['lr_d'],
                        configs['phi_1'],
                        configs['phi_2'],
                        configs['alpha'],
                        configs['beta'],
                        configs['sigma'],
                        configs['gamma'],
                        configs['zeta_1'],
                        configs['zeta_2'],
                        configs['noise_sigma'],
                        use_indicator)
        print("==> Train Done")

    def _pretrain_ae_d(self, batch_size=256, epochs=50, lr_ae=0.0003, lr_d=0.0003,
                      alpha=1., zeta_1=1., zeta_2=1., noise_sigma=[2.5, 1.5], use_indicator=True):
        self.network_ae.train()
        self.network_d.train()
        optimizer_AE = optim.Adam(filter(lambda p: p.requires_grad, self.network_ae.parameters()), lr=lr_ae, amsgrad=False)
        optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, self.network_d.parameters()), lr=lr_d, amsgrad=False)
        train_loader = self.data.get_dataloader(batch_size=batch_size, shuffle=True)
        with tqdm(total=epochs, desc="PreTraining", ncols=100) as pbar:
            for epoch in range(1, epochs + 1):
                epoch_loss_ae = 0.
                epoch_loss_d = 0.
                for _, (x, indicator) in enumerate(train_loader):
                    for i in range(self.view_num):
                        x[i] = x[i].to(self.device)
                        indicator[i] = indicator[i].to(self.device)
                    real_label = []
                    fake_label = []
                    for v in range(self.view_num):
                        real_label.append(Variable(torch.ones(x[v].size(0), 1, device=self.device), requires_grad=False))
                        fake_label.append(Variable(torch.zeros(x[v].size(0), 1, device=self.device), requires_grad=False))

                    ae_loss_list = []
                    d_loss_list = []
                    # -----------------
                    #  Train Autoencoder
                    # -----------------
                    optimizer_AE.zero_grad()

                    rec = self.network_ae.forward_AE(x, noise_sigma)
                    for i in range(self.view_num):
                        # reconstruction loss
                        if use_indicator:
                            recon_loss = self.reconstruction_loss(x[i], rec[i] * indicator[i])
                        else:
                            recon_loss = self.reconstruction_loss(x[i], rec[i])
                        recon_loss *= alpha
                        ae_loss_list.append(recon_loss)

                    if use_indicator:
                        judge = self.network_d.forward_D([rec[0] * indicator[0], rec[1] * indicator[1]])
                    else:
                        judge = self.network_d.forward_D([rec[0], rec[1]])
                    for v in range(self.view_num):
                        # generator ad loss
                        g_adv_loss = self.adversarial_loss(judge[v], real_label[v])
                        g_adv_loss *= zeta_1
                        ae_loss_list.append(g_adv_loss)

                    ae_loss = sum(ae_loss_list)
                    ae_loss.backward()
                    optimizer_AE.step()
                    epoch_loss_ae += ae_loss.item()
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()
                    real = self.network_d.forward_D(x)
                    if use_indicator:
                        fake = self.network_d.forward_D([rec[0].detach() * indicator[0].detach(), rec[1].detach() * indicator[1].detach()])
                    else:
                        fake = self.network_d.forward_D([rec[0].detach(), rec[1].detach()])
                    for v in range(self.view_num):
                        # discriminator ad loss
                        d_adv_loss = self.adversarial_loss(fake[v], fake_label[v]) +\
                                     self.adversarial_loss(real[v], real_label[v])
                        d_adv_loss *= zeta_2
                        d_loss_list.append(d_adv_loss)
                    d_loss = sum(d_loss_list)
                    d_loss.backward()
                    optimizer_D.step()
                    epoch_loss_d += d_loss.item()
                # print('Epoch {}'.format(epoch),
                #       'g_Loss:{:.6f}'.format(epoch_loss_ae / len(train_loader)),
                #       'd_Loss:{:.6f}'.format(epoch_loss_d / len(train_loader)))
                pbar.update(1)
                pbar.set_description('Epoch {}/{} '.format(epoch, epochs) +
                      'g_Loss:{:.6f} '.format(epoch_loss_ae / len(train_loader)) +
                      'd_Loss:{:.6f} '.format(epoch_loss_d / len(train_loader)))

    def _train_ae_d(self, batch_size=256, epochs=200, cutoff=0.5, lr_ae=0.0003, lr_d=0.0003,
              phi_1=.1, phi_2=.1, alpha=1., beta=1., sigma=1., gamma=1., zeta_1=1., zeta_2=1., noise_sigma=[2.5, 2.5], use_indicator=True):
        self.network_ae.train()
        self.network_d.train()
        optimizer_AE = optim.Adam(filter(lambda p: p.requires_grad, self.network_ae.parameters()), lr=lr_ae, amsgrad=False)
        optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, self.network_d.parameters()), lr=lr_d, amsgrad=False)
        train_loader = self.data.get_dataloader(batch_size=batch_size, shuffle=True)
        with tqdm(total=epochs, desc="Training", ncols=100) as pbar:
            for epoch in range(1, epochs + 1):
                epoch_loss_ae = 0.
                epoch_loss_d = 0.
                for _, (x, indicator) in enumerate(train_loader):
                    for i in range(self.view_num):
                        x[i] = x[i].to(self.device)
                        indicator[i] = indicator[i].to(self.device)

                    real_label = []
                    fake_label = []
                    for v in range(self.view_num):
                        real_label.append(Variable(torch.ones(x[v].size(0), 1, device=self.device), requires_grad=False))
                        fake_label.append(Variable(torch.zeros(x[v].size(0), 1, device=self.device), requires_grad=False))

                    ae_loss_list = []
                    d_loss_list = []
                    # -----------------
                    #  Train Autoencoder
                    # -----------------
                    optimizer_AE.zero_grad()

                    z, ls, xrs, zrs = self.network_ae.forward_AEZ(x, noise_sigma)
                    for i in range(self.view_num):
                        # reconstruction loss
                        if use_indicator:
                            recon_loss = self.reconstruction_loss(x[i], xrs[i] * indicator[i]) * phi_1 +\
                                         self.reconstruction_loss(x[i], zrs[i] * indicator[i])
                        else:
                            recon_loss = self.reconstruction_loss(x[i], xrs[i]) * phi_1 + \
                                         self.reconstruction_loss(x[i], zrs[i])
                        recon_loss *= alpha
                        ae_loss_list.append(recon_loss)
                    if use_indicator:
                        fake_single_gen = self.network_d.forward_D([xrs[0] * indicator[0], xrs[1] * indicator[1]])
                        fake_multi_gen = self.network_d.forward_D([zrs[0] * indicator[0], zrs[1] * indicator[1]])
                    else:
                        fake_single_gen = self.network_d.forward_D([xrs[0], xrs[1]])
                        fake_multi_gen = self.network_d.forward_D([zrs[0], zrs[1]])

                    for v in range(self.view_num):
                        # generator ad loss
                        g_adv_loss = self.adversarial_loss(fake_single_gen[v], real_label[v])*phi_2 +\
                                     self.adversarial_loss(fake_multi_gen[v], real_label[v])
                        g_adv_loss *= zeta_1
                        ae_loss_list.append(g_adv_loss)

                    # contrastive loss
                    cls_loss = self.contrastive_loss(*ls) * sigma
                    ae_loss_list.append(cls_loss)

                    if epoch > epochs*cutoff:
                        # add kl_loss and prediction_loss
                        cs = self.network_ae.forward_cross(ls)
                        for i in range(self.view_num):
                            # prediction loss
                            pre_loss = self.mse_loss(ls[i], cs[i])
                            pre_loss *= beta
                            ae_loss_list.append(pre_loss)

                        # cal latent
                        num, lq = self.kld_loss.cal_latent(z)
                        lp = self.kld_loss.target_distribution(lq)
                        lq = lq + torch.diag(torch.diag(num))
                        lp = lp + torch.diag(torch.diag(num))
                        # kl loss
                        kl_loss = self.kld_loss(lp, lq)
                        kl_loss *= gamma
                        ae_loss_list.append(kl_loss)

                    ae_loss = sum(ae_loss_list)
                    ae_loss.backward()
                    optimizer_AE.step()
                    epoch_loss_ae += ae_loss.item()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()
                    if use_indicator:
                        fake_single = self.network_d.forward_D([xrs[0].detach() * indicator[0].detach(), xrs[1].detach() * indicator[1].detach()])
                        fake_multi = self.network_d.forward_D([zrs[0].detach() * indicator[0].detach(), zrs[1].detach() * indicator[1].detach()])
                    else:
                        fake_single = self.network_d.forward_D([xrs[0].detach(), xrs[1].detach()])
                        fake_multi = self.network_d.forward_D([zrs[0].detach(), zrs[1].detach()])
                    real = self.network_d.forward_D(x)

                    for v in range(self.view_num):

                        # discriminator ad loss
                        d_adv_loss = (self.adversarial_loss(fake_single[v], fake_label[v]) +\
                                     self.adversarial_loss(real[v], real_label[v])) * phi_2 +\
                                     (self.adversarial_loss(fake_multi[v], fake_label[v]) +\
                                      self.adversarial_loss(real[v], real_label[v]))
                        d_adv_loss *= zeta_2
                        d_loss_list.append(d_adv_loss)
                    d_loss = sum(d_loss_list)
                    d_loss.backward()
                    optimizer_D.step()
                    epoch_loss_d += d_loss.item()
                # print('Epoch {}'.format(epoch),
                #       'g_Loss:{:.6f}'.format(epoch_loss_ae / len(train_loader)),
                #       'd_Loss:{:.6f}'.format(epoch_loss_d / len(train_loader)))
                pbar.update(1)
                pbar.set_description('Epoch {}/{} '.format(epoch, epochs) +
                      'g_Loss:{:.6f} '.format(epoch_loss_ae / len(train_loader)) +
                      'd_Loss:{:.6f} '.format(epoch_loss_d / len(train_loader)))

    @torch.no_grad()
    def save_latent(self, save_path, Dataname=None):
        if Dataname is None:
            Dataname = self.data.dataname
        self.network_ae.eval()
        x = self.data.get_one_epoch_data(self.device)
        z = self.network_ae.get_common_latent_representation(x)
        kmeans = KMeans(n_clusters=len(np.unique(self.data.get_labels())), n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(z)
        np.savetxt(os.path.join(save_path, '{}_latent_ours.csv'.format(Dataname)), z, delimiter=",")
        np.savetxt(os.path.join(save_path, '{}_pred_ours.csv'.format(Dataname)), y_pred, delimiter=",")

    @torch.no_grad()
    def save_kmeans_pred(self, save_path, Dataname=None):
        if Dataname is None:
            Dataname = self.data.dataname
        self.network_ae.eval()
        x = self.data.get_one_epoch_data(self.device)
        z = self.network_ae.get_common_latent_representation(x)
        kmeans = KMeans(n_clusters=len(np.unique(self.data.get_labels())), n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(z)
        # np.savetxt(os.path.join(save_path, '{}_latent_ours.csv'.format(Dataname)), z, delimiter=",")
        np.savetxt(os.path.join(save_path, '{}_pred_ours.csv'.format(Dataname)), y_pred, delimiter=",")


    @torch.no_grad()
    def test_cluster(self, kmeans_seed=42):
        self.network_ae.eval()
        x = self.data.get_one_epoch_data(self.device)
        Y = self.data.get_labels()
        z = self.network_ae.get_common_latent_representation(x)
        kmeans = KMeans(n_clusters=len(np.unique(Y)), n_init=20, random_state=kmeans_seed)
        y_pred = kmeans.fit_predict(z)
        nmi, ari = eval_tools.clustering_metric(Y, y_pred)
        print('fusion: NMI= %.4f ARI= %.4f' % (nmi, ari))
        return nmi, ari