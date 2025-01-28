from src.combined_processor import CombinedProcessor
from src.eeg_config import EEGConfig
from run.train_VQVAE import train_VQVAE
from src.utils import *

from torch.utils.data import DataLoader, random_split
from moabb.datasets import Lee2019_SSVEP, BI2015b
from torch import save
import matplotlib.pyplot as plt

if __name__ == '__main__':
    lee_config = EEGConfig(
        dataset=Lee2019_SSVEP(),
        num_channels=32,
        sample_rate=1000,
        sample_duration=4,
        subject_range=(1, 5),
        filter_range=(5, 95),
        channel_range=("Fp1", "PO10")
    )
    invaders_config = EEGConfig(
        dataset=BI2015b(),
        num_channels=32,
        sample_rate=512,
        sample_duration=1,
        subject_range=(1, 5),
        filter_range=(5, 95),
        channel_range=("Fp1", "PO10")
    )
    configs = [lee_config, invaders_config]
    n_fft = 512

        
    combined_processor = CombinedProcessor(configs=configs, nperseg = n_fft/2, noverlap = n_fft/4, window='hann')
    train_set, val_set = random_split(combined_processor, [.7, .3])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model, trainer = train_VQVAE(
    dl_train = train_loader,
    dl_val = val_loader,
    epochs = epochs,
    train_bool = True,
    eff_net_PATH = eff_net_PATH,
    classes = [],
    in_channels = lee_config.num_channels)

    train_loss = trainer.logged_metrics['loss_epoch']
    val_loss = trainer.logged_metrics['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    save(model,VQVAE_PATH)