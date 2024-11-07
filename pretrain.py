import os
import numpy as np
from random import shuffle
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score
from models import get_model
import config

def pretrain(dataloaders, device, epochs):
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            epoch_loss = 0.0
            epoch_true, epoch_pred = [], []
            total_samples = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.squeeze(1)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                    
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                epoch_loss += loss.item() * inputs.size(0)

                epoch_true.append(labels.cpu().numpy())
                epoch_pred.append(outputs.max(1)[1].cpu().numpy())

                total_samples += inputs.size(0)

            epoch_true = np.concatenate(epoch_true)
            epoch_pred = np.concatenate(epoch_pred)

            epoch_accuracy = accuracy_score(epoch_true, epoch_pred)
            epoch_loss = epoch_loss / total_samples

            print(f'pretraining {phase} loss: {epoch_loss:.4f}')
            print(f'pretraining {phase} accuracy: {epoch_accuracy:.4f}')

    torch.save(model.state_dict(), os.path.join(config.MODEL_PATH, 'pre_model.pt'))