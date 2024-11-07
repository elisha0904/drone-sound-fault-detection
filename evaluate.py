import os
import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models import get_model 

import config

def scoring(test_true, test_pred):
    accuracy = accuracy_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred, average='macro')
    precision = precision_score(test_true, test_pred, average='macro')
    recall = recall_score(test_true, test_pred, average='macro')

    return (accuracy, f1, precision, recall)

def testing(dataloaders, device):
    non_model = get_model().to(device)
    non_model.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, 'pre_model.pt')))

    adt_model = get_model().to(device)
    adt_model.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, 'target_model.pt')))

    non_model.eval()
    adt_model.eval()

    non_score, adt_score = {}, {}

    for drone in dataloaders.keys():
        non_pred, non_true = [], []
        adt_pred, adt_true = [], []
    
        for inputs, labels in tqdm(dataloaders[drone]['test']):
            inputs = inputs.squeeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            # ---- non-adapted -----
            with torch.no_grad():
                non_output = non_model(inputs)
                non_pred_labels = non_output.max(1)[1]

                non_true.append(labels.cpu().numpy())
                non_pred.append(non_pred_labels.cpu().numpy())

            # ---- adapted -----
            with torch.no_grad():
                adt_output = adt_model(inputs)
                adt_pred_labels = adt_output.max(1)[1]

                adt_true.append(labels.cpu().numpy())
                adt_pred.append(adt_pred_labels.cpu().numpy())

        non_true = np.concatenate(non_true)
        non_pred = np.concatenate(non_pred)

        adt_true = np.concatenate(adt_true)
        adt_pred = np.concatenate(adt_pred)

        non_score[drone] = scoring(non_true, non_pred)
        adt_score[drone] = scoring(adt_true, adt_pred)

    return non_score, adt_score

def print_score(dataloaders, device):
    non_score, adt_score = testing(dataloaders, device)

    for drone in dataloaders.keys():
        non_score_str = ', '.join([f"{x:.3f}" for x in non_score[drone]])
        adt_score_str = ', '.join([f"{x:.3f}" for x in adt_score[drone]])

        print(f"Drone {drone} >>> non : {non_score_str} / adt : {adt_score_str}")

if __name__ == '__main__':
    print_score(dataloaders, config.DEVICE)



