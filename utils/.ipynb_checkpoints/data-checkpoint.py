import torch
import os
from torchvision import transforms, datasets

def make_dataloaders(dataset, batch_size, **kwargs):
    '''
    Build dataloaders for different datasets. The dataloader can be easily iterated on.
    Supports Mnist, FashionMNIST, more to come
    '''

    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'fashionmnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError

    return train_loader, val_loader

def save_model(model, name):
    dir = "./checkpoints"
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir, name)
    torch.save(model, path)
