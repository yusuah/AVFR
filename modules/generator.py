import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork
from modules.audio import AudioEncoder
from modules.attention import AudioVisualAttention


class OcclusionAwareGenerator(nn.Module):
    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False, audio_encoder_params=None, attention_encoder_params=None,
                 motion_channels=256, audio_channels=256, mouth_channels=1, device='cuda'):
        super(OcclusionAwareGenerator, self).__init__()

        self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                       estimate_occlusion_map=estimate_occlusion_map,
                                                       **dense_motion_params)
        self.audio_encoder = AudioEncoder() 
        self.attention_encoder = AudioVisualAttention() 
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = block_expansion * (2 ** i)
            out_features = block_expansion * (2 ** (i + 1))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = block_expansion * (2 ** num_down_blocks)
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        up_blocks = []
       
        enc_dec_channels = motion_channels + audio_channels + mouth_channels
        in_features = in_features + enc_dec_channels
        
        for i in range(num_down_blocks):
            skip_in_features = block_expansion * (2 ** (num_down_blocks - 1 - i))
            out_features = block_expansion * (2 ** (num_down_blocks - 1 - i))
            
            up_blocks.append(UpBlock2d(in_features + skip_in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            in_features = out_features

        self.up_blocks = nn.ModuleList(up_blocks)
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=False)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source, audio_input):
        skips = []
        out = self.first(source_image)
        skips.append(out)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            skips.append(out)

        encoder_bottleneck = self.bottleneck(out)

        dense_motion = self.dense_motion_network(source_image, kp_driving, kp_source)
        audio_feat = self.audio_encoder(audio_input)
        motion_feat = dense_motion['motion_feature_map']
        mouth_feat = self.attention_encoder(audio_input, motion_feat)
      
        target_size = encoder_bottleneck.shape[2:]
        motion_feat = F.interpolate(motion_feat, size=target_size, mode='bilinear', align_corners=False)
        audio_feat = F.interpolate(audio_feat, size=target_size, mode='bilinear', align_corners=False)
        mouth_feat = F.interpolate(mouth_feat, size=target_size, mode='bilinear', align_corners=False)
        enc_dec = torch.cat([motion_feat, audio_feat, mouth_feat], dim=1)

        out = torch.cat([encoder_bottleneck, enc_dec], dim=1)

        for i in range(len(self.up_blocks)):
            skip = skips[len(skips) - 2 - i]
            out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            out = torch.cat([out, skip], dim=1)
            out = self.up_blocks[i](out)

        prediction = torch.sigmoid(self.final(out))

        output_dict = {}
        output_dict["prediction"] = prediction
        output_dict['mask'] = dense_motion['mask']
        output_dict['sparse_deformed'] = dense_motion['sparse_deformed']
        
        deformation_field = dense_motion['deformation']
        output_dict['deformed'] = self.deform_input(source_image, deformation_field)
        output_dict['attention_feature'] = mouth_feat
        output_dict['audio_feature'] = audio_feat
        if 'occlusion_map' in dense_motion:
            output_dict['occlusion_map'] = dense_motion['occlusion_map']

        return output_dict