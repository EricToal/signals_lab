import sys
sys.path.append('../../')

from moabb.datasets import BI2015b, Lee2019_SSVEP
from tqdm.notebook import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# def construct_dataset(eeg_dim=512):
#     ds1 = BI2015b().get_data(subjects=[i + 1 for i in range(5)]) 
#     dstemp= Lee2019_SSVEP()
#     ds2=dstemp.get_data(subjects=[i + 1 for i in range(2)])

#     eegs = []
#     for s in tqdm(range(5)):
#         for j in range(4):
#             df = ds1[s + 1]['0'][f'{j}'].filter(5, 95, verbose=0).to_data_frame().astype(np.float32)
#             segments = np.split(df.loc[:, 'Fp1':'PO10'].values, df.index.stop // eeg_dim) 
#             #segments = np.split(df.loc['Fp1':'PO10',:].values, df.index.stop // eeg_dim) 
#             for segment in segments:
                
#                 eeg = torch.Tensor(segment).repeat(1, 4)
#                 eegs.append(eeg.unsqueeze(0))
                
#     for s in tqdm(range(2)):
#         for j in range(2):
#             df = ds2[s + 1][f'{j}']['1train'].filter(5, 95, verbose=0).to_data_frame().astype(np.float32)
#             segments = np.split(df.loc[:, 'Fp1':'EMG2'].values, df.index.stop // eeg_dim)
#             #segments = np.split(df.loc['Fp1':'EMG2',:].values, df.index.stop // eeg_dim)
#             for segment in segments:
#                 eeg = torch.Tensor(segment).repeat(1, 2)
#                 eegs.append(eeg.unsqueeze(0))

#     eegs = torch.vstack(eegs)
#     torch.save(eegs, './complete_dataset_tensor.pth')
#     return eegs

def construct_dataset(eeg_dim=512):
    ds1 = BI2015b().get_data(subjects=[i + 1 for i in range(5)]) 
    dstemp= Lee2019_SSVEP()
    ds2=dstemp.get_data(subjects=[i + 1 for i in range(2)])

    eegs = []
    for s in tqdm(range(5)):
        for j in range(4):
            df = ds1[s + 1]['0'][f'{j}'].filter(5, 95, verbose=0).to_data_frame().astype(np.float32)
            
            arr1 = df.loc[:, 'Fp1':'PO10'].values
            num_chunks1 = arr1.shape[0] // eeg_dim

            reshaped_arr1 = arr1[:num_chunks1 * eeg_dim].reshape(-1, eeg_dim, 32)
            #reshaped_arr1 = arr1.reshape(-1, eeg_dim, 32)

            for segment in reshaped_arr1:
                
                eeg = torch.Tensor(segment).repeat(1, 4)
                eegs.append(eeg.unsqueeze(0))
              
    for s in tqdm(range(2)):
        for j in range(2):
            df = ds2[s + 1][f'{j}']['1train'].filter(5, 95, verbose=0).to_data_frame().astype(np.float32)
            arr2 = df.loc[:, 'Fp1':'EMG2'].values
            num_chunks2 = arr2.shape[0] // eeg_dim

            reshaped_arr2 = arr2[:num_chunks2 * eeg_dim].reshape(-1, eeg_dim, 64)
            #reshaped_arr2 = arr2.reshape(-1, eeg_dim, 64)
            for segment in reshaped_arr2:
                eeg = torch.Tensor(segment).repeat(1, 2)
                eegs.append(eeg.unsqueeze(0))

    eegs = torch.vstack(eegs)
    torch.save(eegs, './complete_dataset_tensor.pth')
    return eegs



class EegPretrainDataset(Dataset):
    def __init__(self, dim=512, split='train', train_perc=0.7, ds_path=None, ds=None, seed=0):
        self.ds = ds
        self.split = split
        self.train_perc = train_perc
        self.seed = seed
        self.dim = dim
        torch.manual_seed(seed)

        if self.ds is None and ds_path:
            self.ds = torch.load(ds_path)
            print('Dataset loaded from file')

        if self.ds is None:
            self.ds = construct_dataset(dim)
        
        self.train_ds, self.test_ds = self._split(train_perc, split, seed)

    def _split(self, train_perc, split, seed):
        generator = torch.manual_seed(seed)
        
        indices = torch.randperm(len(self.ds), generator=generator)
        train_size = int(len(self.ds) * train_perc)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_ds = self.ds[train_indices]
        test_ds = self.ds[test_indices]

        if split == 'train':
            return train_ds, test_ds
        else:
            return test_ds, train_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]
    
    def __call__(self):
        return self.train_ds, self.test_ds