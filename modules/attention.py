import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEncoder(nn.Module):
    """
    Attention Encoder for transforming I_mel (Mel Spectrogram) into ENC_query.
    Converts (B, 1, 80, 16) → (B, 256, 1, 1)
    """
    def __init__(self):
        super(AttentionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(10, 2), stride=1, padding=0)  

    def forward(self, I_mel):
        x = F.relu(self.conv1(I_mel))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)) 

        return x

class AudioVisualAttention(nn.Module):
    """
    Audio-Visual Attention module.
    Computes an attention map using audio (`ENC_query`) and motion (`ENC_motion`).
    """
    def __init__(self,  input_dim=256, output_dim=1):
        super(AudioVisualAttention, self).__init__()
        self.attention_encoder = AttentionEncoder()
    
    def forward(self, I_mel, ENC_motion):
        """
        I_mel : (B, 1, 80, 16) → Audio spectrogram input
        ENC_motion : (B, 256, 64, 64) → Motion feature map from Dense Motion Network

        Returns:
            ENC_attn : (B, 1, 64, 64) → Attention Map
        """
        ENC_query = self.attention_encoder(I_mel) 
       
        ENC_query = ENC_query.view(ENC_query.shape[0], ENC_query.shape[1], 1, 1)  
       
        ENC_attn = torch.sum(ENC_motion * ENC_query, dim=1, keepdim=True)
     
        ENC_attn = torch.sigmoid(ENC_attn)  
        return ENC_attn
