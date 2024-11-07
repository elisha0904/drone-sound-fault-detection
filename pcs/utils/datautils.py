import os
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

CLASS_MAPPING = {
    'N': 0,
    'MF1': 1, 
    'MF2': 2,
    'MF3': 3, 
    'MF4': 4,
    'PC1': 5,
    'PC2': 6,
    'PC3': 7,
    'PC4': 8
}

class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir,
        drone,
        split,
        sample_rate=16000,
        duration=1,
        transform=None,
        keep_in_mem=False,
        ret_index=False,
    ):
        """
        Args:
            root_dir (str): 데이터셋의 루트 디렉토리 경로
            drone (str): 대상 도메인 (drone 종류)
            split (str): 'train', 'test', 'valid'
            sample_rate (int): 오디오 샘플링 레이트
            duration (float): 오디오 길이(초)
            transform: 오디오에 적용할 변환
            keep_in_mem (bool): 메모리에 데이터 유지 여부
            ret_index (bool): 인덱스 반환 여부
        """
        self.root_dir = os.path.join(root_dir, drone, split)
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.keep_in_mem = keep_in_mem
        self.ret_index = ret_index
        
        # 파일 리스트와 레이블 생성
        self.files = []
        self.labels = []
        
        # 각 클래스 폴더 순회
        for class_name in CLASS_MAPPING.keys():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.wav'):
                    self.files.append(os.path.join(class_dir, file_name))
                    self.labels.append(CLASS_MAPPING[class_name])
        
        # 메모리에 유지할 경우
        self.audio_data = []
        if self.keep_in_mem:
            for file_path in self.files:
                waveform, _ = torchaudio.load(file_path)
                if self.transform:
                    waveform = self.transform(waveform)
                self.audio_data.append(waveform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.keep_in_mem:
            waveform = self.audio_data[idx]
        else:
            waveform, _ = torchaudio.load(self.files[idx])
            if self.transform:
                waveform = self.transform(waveform)
        
        label = self.labels[idx]
        
        if not self.ret_index:
            return waveform, label
        else:
            return idx, waveform, label

def create_dataset(
    domain,
    split="",
    suffix="",
    keep_in_mem=False,
    ret_index=False,
    transform=None,
    sample_rate=16000,
    duration=1
):
    
    if suffix != "":
        suffix = "_" + suffix
    if split == "":
        split = f"{domain}{suffix}"
    
    return AudioDataset(
        root_dir=f"processed_dataset/",
        domain=domain,
        split=split,
        sample_rate=sample_rate,
        duration=duration,
        keep_in_mem=keep_in_mem,
        ret_index=ret_index,
        transform=transform,
    )

def worker_init_seed(worker_id):
    
    np.random.seed(12 + worker_id)
    random.seed(12 + worker_id)

def create_loader(dataset, batch_size=32, num_workers=4, is_train=True):
    
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        num_workers=num_workers,
        shuffle=is_train,
        drop_last=is_train,
        pin_memory=True,
        worker_init_fn=worker_init_seed,
    )

# 기본 전처리
class AudioTransform:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def __call__(self, waveform):
        
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if waveform.size(-1) != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=waveform.size(-1),
                new_freq=self.sample_rate
            )(waveform)
            
        return waveform

def get_class_num(dataset_name):
    return len(CLASS_MAPPING)