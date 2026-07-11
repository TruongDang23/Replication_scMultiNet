import os
import torch
import random
import argparse
import warnings
import numpy as np

from MultiNet.trainer import Trainer
from MultiNet.dataloader import SingleCellMultiOmicsData
from MultiNet.multinet import MultiNet
from MultiNet.configs import *
warnings.filterwarnings("ignore")

DefaultDataname = 'BMNC'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--train', action='store_false', default=True)
parser.add_argument('--dataset', default=DefaultDataname, help='Name of Dataset')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--pretrain_epochs', default=100, type=int, help='Total Epochs of pretrain phase')
parser.add_argument('--train_epochs', default=200, type=int, help='Total Epochs of of train phase')
parser.add_argument('--lr_ae', default=0.0005, type=float, help='leaning_rate of autoencoder')
parser.add_argument('--lr_d', default=0.003, type=float, help='leaning_rate of discriminator')
parser.add_argument('--f1', default=2000, type=int, help='Number of mRNA after feature selection')
parser.add_argument('--f2', default=2000, type=int, help='Number of ADT/ATAC after feature selection')
parser.add_argument('--filter1', action='store_true', default=True, help='Do mRNA selection')
parser.add_argument('--filter2', action='store_true', default=True, help='Do ADT/ATAC selection')
parser.add_argument('--cutoff', default=0.5, type=float, help='CutOff of kl loss and pre loss in training stage')
parser.add_argument('--alpha', default=1, type=float, help='coefficient of the reconstruction loss')
parser.add_argument('--beta', default=0.5, type=float, help='coefficient of the prediction loss in training stage')
parser.add_argument('--phi_1', default=0.005, type=float, help='coefficient of the shared latent features')
parser.add_argument('--phi_2', default=0.005, type=float, help='coefficient of modality-specific features')
parser.add_argument('--sigma', default=1.0, type=float, help='coefficient of the contrastive loss in training stage')
parser.add_argument('--gamma', default=1.0, type=float, help='coefficient of the KL loss in training stage')
parser.add_argument('--zeta_1', default=0.005, type=float, help='coefficient of the adversarial loss of the autoencoder')
parser.add_argument('--zeta_2', default=0.003, type=float, help='coefficient of the adversarial loss of the discriminator')
parser.add_argument('--noise_sigma1', default=2.5, type=float)
parser.add_argument('--noise_sigma2', default=2.5, type=float)
parser.add_argument('--data_path', default=None, type=str)
parser.add_argument('--save_dir', default="./output", type=str)
parser.add_argument('--save', action='store_true', default=None)
parser.add_argument('--mode', default='integrate', type=str)
parser.add_argument('--weight_dir', default='./weights')
parser.add_argument('--weight_name', default=None)
parser.add_argument('--activation1', default='relu', type=str, help='Activation of autoencoders')
parser.add_argument('--activation2', default='leakyrelu', type=str, help='Activation of discriminators')
parser.add_argument('-el1', '--encodeLayer1', nargs='+', default=[256, 64])
parser.add_argument('-el2', '--encodeLayer2', nargs='+', default=[256, 64])
parser.add_argument('--hidden_dim', default=32, type=int, help='Dim of Prediction latent')
parser.add_argument('-dl1', '--decodeLayer1', nargs='+', default=[64, 256])
parser.add_argument('-dl2', '--decodeLayer2', nargs='+', default=[64, 256])
parser.add_argument('-disl1', '--discriminateLayer1', nargs='+', default=[256, 64, 32])
parser.add_argument('-disl2', '--discriminateLayer2', nargs='+', default=[256, 64, 32])
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
# print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = None
if args.dataset in citeseq_list:
    configs = citeseq_config
    print('Use citeseq_config')
elif args.dataset in smageseq_list:
    configs = smageseq_config
    print('Use smageseq_config')
else:
    configs = makeconfigs(args)
    print('Use yourseq_config')
configs['use_indicator'] = True if args.mode == 'imputation' else False
print('use indicator:%s' % (configs['use_indicator']))
print(configs[f'{args.dataset}_Params'])
Dataname = args.dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(args.weight_dir):
    os.mkdir(args.weight_dir)
if args.weight_name is None:
    weight_name = "{}.pth.tar".format(Dataname)
else:
    weight_name = args.weight_name
weight_path = os.path.join(args.weight_dir, weight_name)

if __name__ == "__main__":
    # make data object
    data = SingleCellMultiOmicsData(Dataname, configs, args.data_path)
    # Build network
    multinet = MultiNet(data.n_vars_list, configs)
    multinet = multinet.to(device)
    print(multinet)

    # Build a trainer
    trainer = Trainer(data, multinet, device)
    # Train model for Epochs
    if args.train:
        trainer.train(configs)
    else:
        multinet.load_weight(weight_path)
    if args.save:
        print("==> Network Weights Saving...")
        multinet.save_weight(weight_path=weight_path)
        print("==> saving latent and pred...")
        trainer.save_latent(save_path=args.save_dir)
        print("==> Done")
    trainer.test_cluster()

