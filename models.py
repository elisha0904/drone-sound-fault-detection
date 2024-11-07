import torch
import torch.nn as nn
import torchaudio
from torchvision import models

class CombinedModel(nn.Module):
    def __init__(self, out_dim):  # out_dim 파라미터 추가
        super(CombinedModel, self).__init__()
        self.wav2vec2 = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        self.vgg16 = self._get_modified_vgg16(out_dim)  # out_dim 전달

    def _get_modified_vgg16(self, out_dim):  # out_dim 파라미터 사용
        vgg16 = models.vgg16(pretrained=True)
        modified_layers = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
        vgg16.features = nn.Sequential(*[list(vgg16.features.children())[i] for i in modified_layers])
        vgg16.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 32))
        vgg16.classifier = nn.Sequential(
            nn.Linear(in_features=8192, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=out_dim)  # config.CLASSES 대신 out_dim 사용
        )
        return vgg16

    def forward(self, x):
        att = self.wav2vec2(x)[0]
        vgg_in = att.unsqueeze(1).repeat(1, 3, 1, 1)
        output = self.vgg16(vgg_in)
        return output

def get_model(out_dim):  # out_dim 파라미터 추가
    return CombinedModel(out_dim)