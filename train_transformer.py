import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from src.Transformer import TransformerModel
import matplotlib.pyplot as plt
from src.utils import *
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from src.eeg_config import EEGConfig
from src.combined_processor import CombinedProcessor
from moabb.datasets import Lee2019_SSVEP, BI2015b

def train_model(m, x_train_loader,y_train_loader,x_test_loader,y_test_loader, epochs=10):
    train_loss = []
    validation_loss = []

    for iter in tqdm(range(epochs)):
        m.train()
        total_loss = 0
        for xb,yb in zip(x_train_loader,y_train_loader):
            xb , yb = xb.to(device), yb.to(device)
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss / len(x_train_loader))
        print(f'Epoch {iter + 1}, Train Loss: {total_loss / len(x_train_loader)}')

        m.eval()
        total_loss = 0
        for xb,yb in zip(x_test_loader,y_test_loader):
            xb , yb = xb.to(device), yb.to(device)
            logits, loss = m(xb, yb)
            total_loss += loss.item()
        validation_loss.append(total_loss / len(x_test_loader))
        print(f'Epoch {iter + 1}, Validation Loss: {total_loss / len(x_test_loader)}')

    plt.figure(figsize=(10, 5)) 
    plt.plot(range(1,epochs+1), train_loss, 'bo-', label='Training Loss') 
    plt.plot(range(1,epochs+1), validation_loss, 'ro-', label='Validation Loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.title('Training and Validation Loss over Epochs') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig("TransformerLossCurves.png")
    return m

if __name__ == '__main__':

    #device = 'cuda:0'
    vqvae_model = torch.load(VQVAE_PATH).to(device)
    m = TransformerModel().to(device)
    m = torch.load(TRANSFORMER_MODEL_PATH).to(device)
    
    lee_config = EEGConfig(
        dataset=Lee2019_SSVEP(),
        num_channels=32,
        sample_rate=1000,
        sample_duration=4,
        subject_range=(1, 5),
        filter_range=(5, 95),
        channel_range=("Fp1", "PO4")
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
    combined_processor = CombinedProcessor(configs=configs, subset_labels=True)
    train_set, test_set = random_split(combined_processor, [.7, .3])
    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for x, label in dl_train:
        print(f"x shape: {x.shape}")  # (batch_size, num_channels, eeg_dim)
        print(f"label shape: {label.shape}")  # (batch_size, 1)
        encoded = vqvae_model.codebook.quantize_indices(vqvae_model.encode(x.to(device)))
        print(f"Encoded shape: {encoded.shape}")  # (batch_size, num_tokens)
        break  # Only print for the first batch


    data_label = torch.cat([torch.cat((label.to(device), vqvae_model.codebook.quantize_indices(vqvae_model.encode(x.to(device)))), dim=1) for x, label in dl_train],dim=0)

    
    datasetlen = len(data_label)
    print(data_label[0].shape)
    print(data_label.shape)
    x_train = torch.stack([data_label[i,:block_size] for i in range(len(data_label))])
    y_train = torch.stack([data_label[i,:block_size+1] for i in range(len(data_label))])

    optimizer = torch.optim.Adam(m.parameters(), lr = 1e-3)
    x_train_loader = DataLoader(x_train,batch_size=batch_size)
    y_train_loader = DataLoader(y_train,batch_size=batch_size)
    epochs = 10
    learning_rate = 1e-4

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    test_data = torch.cat([torch.cat((label.to(device), vqvae_model.codebook.quantize_indices(vqvae_model.encode(x.to(device)))), dim=1) for x, label in dl_test],dim=0)
    x_test = torch.stack([test_data[i,:block_size] for i in range(len(test_data))])
    y_test = torch.stack([test_data[i,:block_size+1] for i in range(len(test_data))])
    x_test_loader = DataLoader(x_test,batch_size=batch_size)
    y_test_loader = DataLoader(y_test,batch_size=batch_size)

    print(next(iter(x_test_loader)).shape)

    m = train_model(m ,x_train_loader,y_train_loader,x_test_loader,y_test_loader, epochs)
    torch.save(m,TRANSFORMER_MODEL_PATH)
    #context = torch.tensor([[3],[42],[62]]).to(device=device)
    #print(m.generate(context, max_new_tokens=513))