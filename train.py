import argparse
from statistics import mean
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import MNISTNet

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, writer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    idx = 0
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
            writer.add_scalar('training loss', mean(running_loss), idx)
            idx+=1


def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Nom de l'expérience
    parser.add_argument('--exp_name', type=str, default='MNIST', help='experiment name')
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=32, help='size of each training batch')
    
    # Learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer')
    
    # Nombre d'époques d'entraînement
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    # Parsing des arguments
    args = parser.parse_args()
    
    # Assignation des valeurs des arguments à des variables
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Affichage des valeurs pour vérifier
    print(f"Experiment Name: {exp_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")

    ################## 
    #### Data load
    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    ##################
    ### Train
    net = MNISTNet()
    net = net.to(device)
    writer = SummaryWriter(f'runs/MNIST')
    os.makedirs('weights', exist_ok=True)

    # Chemin du fichier de poids
    weight_path = 'weights/mnist_net.pth'

    # Charger les poids si le fichier existe
    if os.path.isfile(weight_path):
        print(f"Loading weights from {weight_path}")
        net.load_state_dict(torch.load(weight_path, weights_only=True))
    else:
        print(f"No weights found at {weight_path}. Starting training from scratch.")

    optimizer = optim.SGD(net.parameters(), lr = lr)
    train(net, optimizer, trainloader, writer, epochs)
    test_acc = test(net, testloader)
    print(f'Test accuracy: {test_acc:.4f}')
    torch.save(net.state_dict(), 'weights/mnist_net.pth')

        


