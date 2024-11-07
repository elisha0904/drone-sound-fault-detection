import os
import argparse
import random

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision.transforms import Compose
from collections import defaultdict

from tqdm import tqdm
from dataset import AudioDataset
from models import get_model
from adaptation import adapt
from pretrain import pretrain
from evaluate import print_score
import config

def few_shot_sampling(train_dataloader, few_shot_samples_per_class):
    class_indices = defaultdict(list)
    dataset = train_dataloader.dataset
    
    for i, item in enumerate(dataset):
        class_indices[item[1]].append(i)

    selected_indices = []
    
    # 각 클래스별로 샘플링
    for class_idx, indices in tqdm(class_indices.items(), desc="Sampling classes"):
        if len(indices) >= few_shot_samples_per_class:
            class_samples = random.sample(indices, few_shot_samples_per_class)
        else:
            class_samples = indices
        
        selected_indices.extend(class_samples)

    few_shot_train_dataloader = DataLoader(dataset, 
                                           batch_size=config.BATCH_SIZE, 
                                           sampler=SubsetRandomSampler(selected_indices),
                                           num_workers=2)

    return few_shot_train_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', help='Pretrain the model')
    parser.add_argument('--adaptation', action='store_true', help='Domain Adaptation')
    parser.add_argument('--test', action='store_true', help='Test the model')
    
    args = parser.parse_args()

    dataloaders = {drone: {} for drone in config.DRONES}

    for drone in config.DRONES:
        for c in config.CATEGORY:
            dataset_path = os.path.join(config.PATH, drone)
            dataset = AudioDataset(dataset_path, c, classes=config.CLASSES)
            dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
            dataloaders[drone][c] = dataloader
            print(f'{drone} : {c} Data Loaded Successfully')

    source_drone = config.SOURCE_DRONE
    target_drone = config.TARGET_DRONE

    source_dataloader = dataloaders[source_drone]
    target_dataloader = dataloaders[target_drone]

    # 모델 초기화
    model = get_model().to(config.DEVICE)

    # Pretrain 모드
    if args.pretrain:
        print("Starting Pretraining...")
        pretrain(source_dataloader, config.DEVICE, config.EPOCH)
        print("Pretraining completed and model saved.")
    
    # Adaptation 모드
    if args.adaptation:
        print("Loading Pretrained model...")
        model.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, 'pre_model.pt')))
        
        print("Performing Few-shot sampling...")
        few_shot_train_dataloader = few_shot_sampling(target_dataloader['train'], config.FEW_SHOT_SAMPLES_PER_CLASS)
        
        print("Starting Domain Adaptation...")
        adapt(model, few_shot_train_dataloader, target_dataloader['valid'], config.DEVICE, config.ADAPT_EPOCH, config.LOSS_TYPE)
        print("Domain adaptation completed and model saved.")

    # Test 모드
    if args.test:
        print("Starting evaluation...")
        print_score(model, dataloaders, config.DEVICE)

if __name__ == "__main__":
    main()

