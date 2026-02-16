import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio.functional as F
import csv
from pathlib import Path
import math

from torch import nn
from .transformer import RoFormer


class BeatModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        config_dataset = config['dataset']
        param = config['model']
        n_mels = config_dataset['n_mels']
        embed_size = param['embed_size']
        spectrogram_size = n_mels * len(config_dataset['window_sizes'])
        n_heads = embed_size // param['head_dim']
        max_seq_length = config_dataset['sample_rate'] * config_dataset['input_size_seconds'] // config_dataset['hop_size']  + 10


        self.encoder = nn.Sequential(
            nn.Conv1d(spectrogram_size, spectrogram_size, 3, padding = 1),
            nn.GELU()
        )
        self.linear = nn.Sequential(
            nn.Linear(spectrogram_size, embed_size),
            nn.GELU()
        )
        self.roformer = RoFormer(
            embed_size, param['n_layers'], n_heads,
            param['mlp_hidden_dim'], max_seq_length
        )
        self.output = nn.Linear(embed_size, 3)
        
    def forward(self, x):
        B, T, F, C = x.shape
        x = x.view(B, T, F * C)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.roformer(x)
        x = self.output(x)
        return x