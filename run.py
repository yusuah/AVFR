import matplotlib

matplotlib.use('Agg')

import inspect
import os, sys
import logging
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

project_root = os.path.dirname(os.path.abspath(__file__))
hrnet_lib_path = os.path.join(project_root, 'HRNet-Facial-Landmark-Detection')
if hrnet_lib_path not in sys.path:
    sys.path.insert(0, hrnet_lib_path)

from modules.frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint import KPDetector
from modules.audio import AudioEncoder
from modules.attention import AudioVisualAttention
import torch

from train import train
from reconstruction import reconstruction
from animate import animate

from lib.models.hrnet import get_face_alignment_net
from lib.config.defaults import _C as cfg
from modules.keypoint import KPDetector 
import torch

if len(sys.argv) == 1:
    sys.argv = [
        "run.py",
        "--config", "/source/sua/AVFR/AVFR/first-order-model/config/vox-256.yaml", 
        "--mode", "train"
    ]

hrnet_config_path = "/source/sua/AVFR/AVFR/first-order-model/config/face_alignment_cofw_hrnet_w18.yaml"
cfg.merge_from_file(hrnet_config_path)
cfg.MODEL.PRETRAINED = "checkpoints/HR18-COFW.pth"

hrnet_model = get_face_alignment_net(cfg, is_train=False)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
    os.environ['GLOG_minloglevel'] = '3'  
    logging.getLogger().setLevel(logging.ERROR)
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    if torch.cuda.is_available() and opt.device_ids:
        device = torch.device('cuda:%d' % opt.device_ids[0])
        print("Using GPU:", device)
    else:
        device = torch.device('cpu')
        print("Using CPU")
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += '_' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    audio_encoder = AudioEncoder()
    attention_encoder = AudioVisualAttention()
    kp_detector_params = config['model_params']['kp_detector_params'].copy()
    
    kp_detector_params['hrnet_model'] = hrnet_model
    kp_detector_params['num_kp'] = cfg.MODEL.NUM_JOINTS
    kp_detector_params['hrnet_out_channels'] = 270 
    kp_detector_params['device'] = device
    
   
    generator_params = config['model_params']['generator_params'].copy()
    generator_params.update(config['model_params']['common_params']) 
    generator_params['num_kp'] = cfg.MODEL.NUM_JOINTS

    kp_detector = KPDetector(**kp_detector_params)
    generator = OcclusionAwareGenerator(**generator_params)
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    
    generator.to(device)
    discriminator.to(device)
    kp_detector.to(device)
    audio_encoder.to(device)
    attention_encoder.to(device)

    if opt.verbose:
        print("Generator:", next(generator.parameters()).device)
        print("Discriminator:", next(discriminator.parameters()).device)
        print("KPDetector:", next(kp_detector.parameters()).device)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])



    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector,opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator,kp_detector, opt.checkpoint, log_dir, dataset)
