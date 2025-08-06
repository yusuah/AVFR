from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian

class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))
        self.conv_final = nn.Conv2d(172, 256, kernel_size=3, padding=1)  
        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(self.scale_factor, num_channels) 
        self.extract_motion_features = nn.Sequential(
           
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((64, 64)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )
    def deform_input(self, inp, deformation): #deformation정보에 따라 이미지 변형하는 함수
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation[:, -2:, :, :]
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)
    
#driving keypoint와 source keypoint (동일한 keypoint 위치)의 위치 차이 히트맵 생성 = 해당 pixel이 어떤 변형을  학습해야하는지 정보 제공
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        spatial_size = source_image.shape[2:] #이미지의 공간적 크기 가져옴
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance) #keypoint를 gaussian heatmap으로 변환
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source #움직임 heatmap 만들기

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()) #배경에 대한 0 히트맵
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

# keypoint 기반 jacobian 변형을 수행하여 픽셀 단위의 변형 행렬 생성 = sparse motion field 생성
    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type()).to(source_image.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2) # [-1,1]좌표 그리드 생성
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2) # driving keypoint 기준으로 (source)좌표 정렬!!!
        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2) #변형된 좌표를 다시 source keypoint 기준으로 정렬

        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)#기본 좌표 격자와 변환된 좌표 격자 합쳐서 sparse 움직임 tensor 만들기
        return sparse_motions

#sparse_motion(픽셀 이동 정보)를 이용해 source image 변형
    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))#각 keypoint 채널별로 변형된 이미지 묶음
        return sparse_deformed

#변형 map 생성
    def forward(self, source_image, kp_driving, kp_source):
        kp_driving['value'] = kp_driving['value'].to(source_image.device)
        kp_source['value'] = kp_source['value'].to(source_image.device)
        bs, _, h, w = source_image.shape
    
        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source) #keypoint마다 이동 벡터를 저장한 2d flow field
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion) #keypoint가 이동한만큼 source image를 변형하여 deformed_source 생성
        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)
        prediction = self.hourglass(input) #hourglass = motion feature map 
        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3) #source image에서 keypoint가 이동한 위치를 담은 2d flow field
        
        #mask를 sparse_motion과 곱하여 최종 dense flow(deformation field)계산
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1) 
        deformed_image = self.deform_input(source_image, deformation)  
        motion_feature_map = self.extract_motion_features(deformed_image) 
        out_dict['deformation'] = deformation #최종 dense flow저장
        out_dict['motion_feature_map'] = motion_feature_map
        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
        #return prediction

        
        #motion field -> dense flow (source와 driving의 위치 차이)->
        #source image에 위치 차이 적용하여 driving motion을 반영한 이미지 생성 -> motion feature map (우리가 만든 이미지로부터 특징 추출)
        