import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import librosa
from modules.frames_dataset import FramesDataset

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=256):
        super(AudioEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(64,64))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

    def forward(self, audio_input):
        device = next(self.parameters()).device 
        audio_input = audio_input.to(device)

        x = F.relu(self.conv1(audio_input))   

        x = F.relu(self.conv2(x))    

        x = self.upsample(x) 
        x = F.relu(self.conv3(x))  
        return x   