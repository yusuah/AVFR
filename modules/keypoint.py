import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from lib.core.evaluation import decode_preds
from modules.util import AntiAliasInterpolation2d
import matplotlib.pyplot as plt
import os
class KPDetector(nn.Module):
    def __init__(self, hrnet_model, num_kp, hrnet_out_channels,
                 input_res=(256, 256), output_res=(64, 64),scale_factor=1,
                 estimate_jacobian=True, pad=3, device='cuda'):
        super(KPDetector, self).__init__()
        self.hrnet = hrnet_model.to(device)
        self.input_res = input_res
        self.output_res = output_res
        self.device = device
        self.num_kp = num_kp
        self.estimate_jacobian = estimate_jacobian
        self.jacobian = None

        if estimate_jacobian:
            self.jacobian = nn.Conv2d(
                in_channels=hrnet_out_channels,
                out_channels=4 * num_kp,
                kernel_size=7,
                padding=pad 
            )
            self.jacobian.weight.data.zero_()
            identity = [1, 0, 0, 1] * num_kp
            self.jacobian.bias.data.copy_(torch.tensor(identity, dtype=torch.float))
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(self.scale_factor, channels=hrnet_out_channels)

    def forward(self, x , center, scale):
  
        input_tensor = x.to(self.device)
        feature_map = self.hrnet(input_tensor, return_feature_map=True)

        if self.scale_factor != 1:
            feature_map = self.down(feature_map)
        heatmap = self.hrnet.head(feature_map)  
      
        os.makedirs('heatmap_vis', exist_ok=True)
        input_img_np = x[0].cpu().numpy().transpose(1, 2, 0) 
        plt.imsave('heatmap_vis/input_image.png', (input_img_np * 255).astype(np.uint8))
        combined_heatmap = torch.sum(heatmap[0], dim=0).detach().cpu().numpy()
        plt.imsave('heatmap_vis/combined_heatmap.png', combined_heatmap, cmap='hot')

        B, K, H, W = heatmap.shape
        kp = decode_preds(heatmap, center, scale, self.output_res).squeeze(0)

        out = {'value': kp}

        if self.estimate_jacobian and self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.view(B, K, 4, H, W)
            heatmap_softmax = F.softmax(heatmap, dim=1).unsqueeze(2) 
            weighted = heatmap_softmax * jacobian_map
            jacobian = weighted.view(B, K, 4, -1).sum(dim=-1)
            jacobian = jacobian.view(B, K, 2, 2)
            out['jacobian'] = jacobian.squeeze(0)

        return out

