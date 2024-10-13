import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # La première convolution prend en entrée 1 canal (image en niveaux de gris)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # Conv1: 32 filtres 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # Conv2: 64 filtres 3x3
        
        # Pooling pour réduire la taille des cartes de caractéristiques
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # La taille de l'entrée de fc1 dépend de la taille des feature maps après les convolutions et pooling
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Après convs/pooling, les cartes sont 5x5 si l'entrée est 28x28
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 sorties pour les 10 classes (chiffres de 0 à 9)

    def forward(self, x):
        x = F.relu(self.conv1(x))       
        x = self.pool(x)               
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 64 * 5 * 5)  # Aplatir les cartes de 64 canaux, taille 5x5 après les convolutions et pooling

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

