import os
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "selu":
            net.append(nn.SELU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "leakyrelu":
            net.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "elu":
            net.append(nn.ELU())
    return nn.Sequential(*net)

def buildNetwork2(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "selu":
            net.append(nn.SELU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "leakyrelu":
            net.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "elu":
            net.append(nn.ELU())
    net.append(nn.Softmax(dim=1))
    return nn.Sequential(*net)

def buildNetwork3(layers, activation="leakyrelu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "selu":
            net.append(nn.SELU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "leakyrelu":
            net.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "elu":
            net.append(nn.ELU())
    net.append(nn.Dropout(0.1))
    net.append(nn.Linear(layers[-1], 1))
    net.append(nn.BatchNorm1d(1, affine=True))
    net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class MultiNet(nn.Module):
    def __init__(self,
                 input_size: list,
                 configs: dict):
        super(MultiNet, self).__init__()
        self.ae_network = NetworkAE(input_size, configs['encoder_dim'],
                                    configs['decoder_dim'], configs['hidden_dim'], configs['activation1'])
        self.d_network = NetworkD(input_size, configs['discriminator_dim'], configs['activation2'])

    def to(self, device):
        self.ae_network = self.ae_network.to(device)
        self.d_network = self.d_network.to(device)
        return self

    def load_weight(self, weight_path):
        if os.path.isfile(weight_path):
            print("==> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path)
            if self.ae_network is not None:
                self.ae_network.load_state_dict(checkpoint['state_ae_dict'])
            if self.d_network is not None:
                self.d_network.load_state_dict(checkpoint['state_d_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(weight_path))
            raise ValueError

    def save_weight(self, weight_path):
        if weight_path is not None:
            print("==> saving checkpoint '{}'".format(weight_path))
            torch.save({'state_ae_dict': self.ae_network.state_dict(),
                        'state_d_dict': self.d_network.state_dict()}, weight_path)
        else:
            print("==> the weigth path:'{}' is not exists".format(weight_path))
            raise ValueError


class Attention(nn.Module):
    def __init__(self,
                n_hidden: int = 32,
                activation='relu',
                n_heads: int = 8,
                dropout_rate: float = 0.1
                ):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(n_hidden, n_hidden)
        self.w_k = nn.Linear(n_hidden, n_hidden)
        self.w_v = nn.Linear(n_hidden, n_hidden)
        self.do = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(n_hidden, eps=0.0001)
        assert n_hidden % 2 == 0 and n_hidden != 0, "Multi-Encoder dims  can't be divided by two"
        self._encoder = buildNetwork([n_hidden, n_hidden // 2], activation)

    def forward(self, X):
        q = torch.cat((X[0], X[1]), 1)

        assert q.shape[1] % self.n_heads == 0, "n_heads can't be divided by seq length!"
        Q = self.w_q(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        K = self.w_k(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        V = self.w_v(q).view(q.shape[0], self.n_heads, q.shape[1] // self.n_heads, -1)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        attention = self.do(torch.softmax(energy, dim=-1))
        q = torch.matmul(attention, V).view(q.shape[0], q.shape[1])
        q = self.norm(q)
        z = self._encoder(q)

        return z


class Prediction(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 activation='relu',
                 hidden_idm=32):
        super(Prediction, self).__init__()
        self._encoder = buildNetwork([input_dim, hidden_idm], activation)
        self._decoder = buildNetwork2([hidden_idm, output_dim], activation)

    def forward(self, X):
        z = self._encoder(X)
        Xr = self._decoder(z)
        return Xr


class NetworkD(nn.Module):
    def __init__(self,
                 input_size,
                 discriminator_dim: list = [[256, 128, 64], [256, 128, 64]],
                 activation: str = 'relu'):
        super(NetworkD, self).__init__()
        self.view_num = 2
        discriminators = []
        for i in range(self.view_num):
            discriminators.append(buildNetwork3([input_size[i]] + discriminator_dim[i], activation))
        self.discriminators = nn.ModuleList(discriminators)


    def forward_D(self, xrs):
        ds = []
        for v in range(self.view_num):
            x = xrs[v]
            d = self.discriminators[v](x)
            ds.append(d)

        return ds


class NetworkAE(nn.Module):
    def __init__(self,
                 input_size: list,
                 encoder_dim: list = [[256, 128, 64], [256, 128, 64]],
                 decoder_dim: list = [[64, 128, 256], [64, 128, 256]],
                 hidden_dim: int = 32,
                 activation: str = 'relu',
                 ):
        super(NetworkAE, self).__init__()

        # Parameters
        self.view_num = len(encoder_dim)

        encoders = []
        decoders = []
        encoder_total_dim = 0
        for i in range(self.view_num):
            encoders.append(buildNetwork2([input_size[i]] + encoder_dim[i], activation))
            decoders.append(buildNetwork(decoder_dim[i] + [input_size[i]], activation))
            encoder_total_dim += encoder_dim[i][-1]

        self.encoders = nn.ModuleList(encoders)
        self.RNA2ADT = Prediction(encoder_dim[0][-1], encoder_dim[1][-1], activation, hidden_dim)
        self.ADT2RNA = Prediction(encoder_dim[1][-1], encoder_dim[0][-1], activation, hidden_dim)
        self.decoders = nn.ModuleList(decoders)
        self.joint_attention_module = Attention(encoder_total_dim)

    @torch.no_grad()
    def get_common_latent_representation(self, X):
        self.eval()
        latents = self.encode(X)
        latent = self.joint_attention_module(latents)
        return latent.detach().cpu().numpy()

    @torch.no_grad()
    def get_denoised_output(self, X):
        self.eval()
        res = []
        latents = self.encode(X)
        latent = self.joint_attention_module(latents)
        zrs = self.decode_multi(latent)
        for i in zrs:
            res.append(i.detach().cpu().numpy())
        return res

    @torch.no_grad()
    def get_single_denoised_output(self, X):
        self.eval()
        res = []
        latents = self.encode(X)
        xrs = self.decode_single(latents)
        for i in xrs:
            res.append(i.detach().cpu().numpy())
        return res

    @torch.no_grad()
    def get_single_latent_representation(self, X):
        self.eval()
        res = []
        latents = self.encode(X)
        for i in latents:
            res.append(i.detach().cpu().numpy())
        return res

    @torch.no_grad()
    def get_cross_modality_RNA2ADT(self, x):
        self.eval()
        RNA_low = self.encoders[0](x)
        pred_RNA = self.RNA2ADT(RNA_low)
        h = self.decoders[1](pred_RNA)
        return h.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_cross_modality_RNA2ATAC(self, x, binarization=True):
        self.eval()
        RNA_low = self.encoders[0](x)
        pred_RNA = self.RNA2ADT(RNA_low)
        h = self.decoders[1](pred_RNA)

        h_np = h.detach().cpu().numpy()  # Convert to numpy array

        def binarizationfunc(imputed, raw):
            # Calculate the row and column means of the raw matrix
            row_means = raw.mean(axis=1)  # Row means
            col_means = raw.mean(axis=0)  # Column means

            # Compare each element and perform logical AND operation
            binarized_matrix = (imputed > row_means[:, np.newaxis]) & (imputed > col_means[np.newaxis, :])

            # Convert to int8 type
            return binarized_matrix.astype(np.int8)

        if binarization:
            h_np = binarizationfunc(h_np, h_np)  # Apply binarization function if binarization is True

        return h_np

    @torch.no_grad()
    def get_cross_modality_ADT2RNA(self, x):
        self.eval()
        ADT_low = self.encoders[1](x)
        pred_ADT = self.ADT2RNA(ADT_low)
        h = self.decoders[0](pred_ADT)
        return h.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_cross_modality_ATAC2RNA(self, x):
        return self.get_cross_modality_ADT2RNA(x)
    
    def encode(self, X):
        latents = []
        for i in range(self.view_num):
            latent = self.encoders[i](X[i])
            latents.append(latent)

        return latents

    def decode_single(self, Z):
        hs = []
        for i in range(self.view_num):
            hs.append(self.decoders[i](Z[i]))
        return hs

    def decode_multi(self, Z):
        hs = []
        for i in range(self.view_num):
            hs.append(self.decoders[i](Z))
        return hs

    def forward_AE(self, X, noise_sigma=[2.5, 2.5]):
        X_noised = []
        for i in range(self.view_num):
            X_noised.append(X[i] + torch.randn_like(X[i])*noise_sigma[i])
        ls = self.encode(X_noised)
        xrs = self.decode_single(ls)
        return xrs

    def forward_AEZ(self, X, noise_sigma=[2.5, 2.5]):
        X_noised = []
        for i in range(self.view_num):
            X_noised.append(X[i] + torch.randn_like(X[i])*noise_sigma[i])

        ls = self.encode(X_noised)
        xrs = self.decode_single(ls)

        z = self.joint_attention_module(ls)
        zrs = self.decode_multi(z)

        clean_ls = self.encode(X)
        clean_z = self.joint_attention_module(clean_ls)
        return clean_z, ls, xrs, zrs

    def forward_cross(self, ls):
        cs = []
        cs.append(self.ADT2RNA(ls[1]))
        cs.append(self.RNA2ADT(ls[0]))
        return cs


