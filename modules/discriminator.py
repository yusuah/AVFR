from torch import nn
import torch.nn.functional as F
from modules.util import kp2gaussian
import torch

class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()

        self.use_kp = use_kp
        self.kp_variance = kp_variance

        input_channels = num_channels + num_kp * use_kp + 256 + 1  

        down_blocks = []
        for i in range(num_blocks):
            in_feats = input_channels if i == 0 else min(max_features, block_expansion * (2 ** i))
            out_feats = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(
                DownBlock2d(in_feats, out_feats, norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x, kp=None, audio_feature=None, attention_feature=None):
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            x = torch.cat([x, heatmap], dim=1)
        target_size = x.shape[2:]
        audio_feature = F.interpolate(audio_feature, size=target_size, mode='bilinear', align_corners=False)
        attention_feature = F.interpolate(attention_feature, size=target_size, mode='bilinear', align_corners=False)
        x = torch.cat([x, audio_feature, attention_feature], dim=1)

        feature_maps = []
        for down_block in self.down_blocks:
            x = down_block(x)
            feature_maps.append(x)

        prediction_map = self.conv(x)
        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        self.discs = nn.ModuleDict({
            str(scale).replace('.', '-'): Discriminator(**kwargs)
            for scale in scales
        })

    def forward(self, x, kp=None, audio_feature=None, attention_feature=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale_key = str(scale).replace('-', '.')
            input_x = x['prediction_' + scale_key]
            feature_maps, prediction_map = disc(input_x, kp, audio_feature, attention_feature)
            out_dict['feature_maps_' + scale_key] = feature_maps
            out_dict['prediction_map_' + scale_key] = prediction_map
        return out_dict
