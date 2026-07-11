import torch
import h5py
import numpy as np
import scanpy as sc
import anndata as ad
from MultiNet.preprocess import read_dataset, normalize, geneSelection
from torch.utils.data import Dataset, DataLoader
from MultiNet.eval_tools import dropout


class Multi_Omics(Dataset):
    def __init__(self, X1, X2, X1_indicator, X2_indicator):
        self.x1 = X1
        self.x2 = X2
        self.x1_indicator = X1_indicator
        self.x2_indicator = X2_indicator

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], \
               [torch.from_numpy(self.x1_indicator[idx]), torch.from_numpy(self.x2_indicator[idx])]


class BaseData:
    def __init__(self, Dataname, configs, datapath, preprocess=True):
        self.dataname = Dataname
        self.datapath = self.get_datapath(Dataname, datapath)
        self.view_num = 2
        self.x1 = None
        self.x2 = None
        x1, x2 = self.read_h5(self.datapath)
        self.importantGenes = None
        self.importantFeatures = None
        self.adata1 = None
        self.adata2 = None
        self.n_vars_list = None
        if preprocess:
            filter1, filter2, f1, f2 = configs[f'{Dataname}_Params']["data_preprocess"]
            x1, x2, self.importantGenes, self.importantFeatures = self.select(x1, x2, filter1, filter2, f1, f2)
            self.adata1, self.adata2, self.n_vars_list = self.data_normalize(x1, x2)
        else:
            print('Warning! Your data has not been pre-processed!\nSaving Raw data to (x1, x2)...')
            self.x1 = x1
            self.x2 = x2
            
    def get_labels(self):
        with h5py.File(self.datapath) as mat:
            Y = mat['Y'].asstr()[:] if h5py.check_string_dtype(mat['Y'].dtype) else np.array(mat['Y'])
        return Y

    def get_celltypes(self):
        with h5py.File(self.datapath) as mat:
            cell_types = mat['Celltypes'].asstr()[:] if h5py.check_string_dtype(mat['Celltypes'].dtype) else np.array(mat['Celltypes'])
        return cell_types

    def get_genes(self, gene='Genes'):
        with h5py.File(self.datapath) as mat:
            gene = np.array(mat[gene])
        if self.importantGenes is not None:
            gene = gene[self.importantGenes]
        return gene

    def get_features(self, feature='ADTS'):
        with h5py.File(self.datapath) as mat:
            f = np.array(mat[feature])
        if self.importantFeatures is not None:
            f = f[self.importantFeatures]
        return f

    def get_dataloader(self, batch_size=256, shuffle=True, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_one_epoch_data(self, device):
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def read_h5(data_path):
        data_mat = h5py.File(data_path)
        x1 = np.array(data_mat['X1'])
        x2 = np.array(data_mat['X2'])
        # y = np.array(data_mat['Y'])
        data_mat.close()
        return x1, x2

    @staticmethod
    def select(x1, x2, filter1, filter2, f1, f2):
        # processing
        importantGenes = None
        importantFeatures = None
        if filter1:
            importantGenes = geneSelection(x1, n=f1, plot=False)
            x1 = x1[:, importantGenes]
        if filter2:
            importantFeatures = geneSelection(x2, n=f2, plot=False)
            x2 = x2[:, importantFeatures]

        return x1, x2, importantGenes, importantFeatures

    @staticmethod
    def data_normalize(x1, x2):
        adata1 = sc.AnnData(x1.astype(np.float32))
        adata1 = read_dataset(adata1,
                              transpose=False,
                              test_split=False,
                              copy=True,
                              fname='genes')
        adata1 = normalize(adata1,
                           filter_min_counts=True,
                           size_factors=True,
                           normalize_input=False,
                           logtrans_input=True)
        adata1 = BaseData.make_indicators(adata1)

        adata2 = sc.AnnData(x2.astype(np.float32))
        adata2 = read_dataset(adata2,
                              transpose=False,
                              test_split=False,
                              copy=True,
                              fname='proteins/peaks')
        adata2 = normalize(adata2,
                           filter_min_counts=True,
                           size_factors=True,
                           normalize_input=False,
                           logtrans_input=True)
        adata2 = BaseData.make_indicators(adata2)

        input_size = [adata1.n_vars, adata2.n_vars]

        return adata1, adata2, input_size

    @staticmethod
    def get_datapath(dataset, datapath):
        if datapath is not None:
            data_path = datapath
        else:
            if dataset == 'PBMC_Spector':
                data_path = './data/CITESeq_pbmc_spector_all.h5'
            elif dataset == 'PBMC3K':
                data_path = './data/SMAGESeq_10X_pbmc_3k_granulocyte_plus.h5'
            elif dataset == 'PBMC10K':
                data_path = './data/SMAGESeq_10X_pbmc_10k_granulocyte_plus.h5'
            elif dataset == 'SLN111D1':
                data_path = './data/CITESeq_spleen_lymph_sln111_d1.h5'
            elif dataset == 'SLN208D1':
                data_path = './data/CITESeq_spleen_lymph_sln208_d1.h5'
            elif dataset == 'BMNC':
                data_path = './data/CITESeq_GSE128639_BMNC_annodata.h5'
            elif dataset == 'COVID19':
                data_path = './data/CITESeq_covid19_pbmc_filtered.h5'
            elif dataset == 'SimulatedData':
                data_path = './data/CITESeq_symsim_simulated_data.h5'
            else:
                raise NotImplementedError

        return data_path

    @staticmethod
    def make_indicators(adata):
        adata.obsm['indicator'] = np.where(adata.X > 0, 1, 0).astype(np.int32)
        return adata


class SingleCellMultiOmicsData(BaseData):
    def __init__(self, Dataname, configs, datapath=None, preprocess=True):
        super(SingleCellMultiOmicsData, self).__init__(Dataname, configs, datapath, preprocess)
        self.dataset = None
        self.data_size = None
        if preprocess:
            self.make_dataset()
        else:
            print('Warning! No training dataset was generated!')
            
    def regenerated_data(self, Genes, Features, feature_name='GeneFromPeaks'):
        rawGenes = self.get_genes().astype(str)
        rawFeatures = self.get_features(feature_name).astype(str)

        index_map_genes = {value: index for index, value in enumerate(rawGenes)}
        importantGenes = np.array([index_map_genes[element] for element in Genes])
        
        index_map_features = {value: index for index, value in enumerate(rawFeatures)}
        importantFeatures = np.array([index_map_features[element] for element in Features])
        
        self.x1 = self.x1[:, importantGenes]
        self.x2 = self.x2[:, importantFeatures]
        self.importantGenes, self.importantFeatures = importantGenes, importantFeatures
        self.adata1, self.adata2, self.n_vars_list = self.data_normalize(self.x1, self.x2)
        self.make_dataset()
        print('Warning! Training dataset regenerated successfully!')
    
    def change_training_feature_value(self, data):
        adata2 = ad.AnnData(X=data)
        adata2 = BaseData.make_indicators(adata2)
        self.adata2 = adata2
        self.make_dataset()
        
    def make_dataset(self):
        self.dataset =\
            Multi_Omics(self.adata1.X, self.adata2.X, self.adata1.obsm['indicator'], self.adata2.obsm['indicator'])
        self.data_size = self.adata1.X.shape[0]
        
    def get_dataloader(self, batch_size=256, shuffle=True, **kwargs):
        train_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader

    def get_one_epoch_data(self, device):
        data_loader = DataLoader(dataset=self.dataset, batch_size=self.data_size, shuffle=False)
        data = []
        for _, (x, indicator) in enumerate(data_loader):
            for i in range(self.view_num):
                x[i] = x[i].to(device)
                data.append(x[i])
        return data


class SingleCellMultiOmicsData_ForImputedTest(BaseData):
    def __init__(self, Dataname, configs, datapath=None, rate=0.1):
        super(SingleCellMultiOmicsData_ForImputedTest, self).__init__(Dataname, configs, datapath, True)
        x1_zero, i1, j1, i1x = dropout(self.adata1.X, rate)
        x2_zero, i2, j2, i2x = dropout(self.adata2.X, rate)
        # save original matrix and index
        self.dropout_dict = {'x1': [self.adata1.X, i1, j1, i1x],
                             'x2': [self.adata2.X, i2, j2, i2x]}

        self.adata1 = ad.AnnData(X=x1_zero)
        self.adata1 = self.make_indicators(self.adata1)
        self.adata2 = ad.AnnData(X=x2_zero)
        self.adata2 = self.make_indicators(self.adata2)
        self.dataset =\
            Multi_Omics(self.adata1.X, self.adata2.X, self.adata1.obsm['indicator'], self.adata2.obsm['indicator'])
        self.data_size = self.adata1.X.shape[0]

    @staticmethod
    def imputed_matrix(X_dopout, X_res, X_raw, i, j, ix):
        all_index = i[ix], j[ix]
        X = X_dopout.copy()
        X[all_index] = X_res[all_index]
        return X

    def get_imputed_matrix(self, X_res):
        X_imputed = []
        X_imputed.append(self.imputed_matrix(self.adata1.X, X_res[0], *self.dropout_dict['x1']))
        X_imputed.append(self.imputed_matrix(self.adata2.X, X_res[1], *self.dropout_dict['x2']))
        return X_imputed

    def get_dataloader(self, batch_size=256, shuffle=True, **kwargs):
        train_loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader

    def get_one_epoch_data(self, device):
        data_loader = DataLoader(dataset=self.dataset, batch_size=self.data_size, shuffle=False)
        data = []
        for _, (x, indicator) in enumerate(data_loader):
            for i in range(self.view_num):
                x[i] = x[i].to(device)
                data.append(x[i])
        return data



