import os
import torch
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from modules.augmentation import AllAugmentationTransform
import glob

import mediapipe as mp
from mediapipe.tasks.python.core import base_options as mp_core_options
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker

import librosa
import librosa.display
import imageio 
import cv2
def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """
    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)
        video_array = np.moveaxis(image, 1, 0)
        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
        
    elif name.lower().endswith(('.gif', '.mp4', '.mov')):
        try:
            reader = imageio.get_reader(name)
        except Exception as e:
            print(f"Error reading video file: {name}")
            raise e
            
        video_frames = []
        try:
            for frame in reader:
                if len(frame.shape) == 2:
                    frame = gray2rgb(frame)
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                video_frames.append(frame)
        finally:
            reader.close()
        
        video_array = img_as_float32(np.array(video_frames))
        
    else:
        raise Exception("Unknown file extensions %s" % name)

    return video_array

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, audio_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir #data/voxpng
        self.audio_dir = audio_dir #data/acc
        #self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            train_videos = []
            train_path = os.path.join(root_dir, 'train')
            for dirpath, dirnames, filenames in os.walk(train_path):
                for filename in filenames:
                    if filename.lower().endswith('.mp4'):
                        train_videos.append(os.path.join(dirpath, filename))
            test_videos = []
            test_path = os.path.join(root_dir, 'test')
            for dirpath, dirnames, filenames in os.walk(test_path):
                for filename in filenames:
                    if filename.lower().endswith('.mp4'):
                        test_videos.append(os.path.join(dirpath, filename))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            #train mode면, root_dir = mp4/train/
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.videos = self.videos[:1000]
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

        landmarker_options = face_landmarker.FaceLandmarkerOptions(
            base_options=mp_core_options.BaseOptions(model_asset_path='/source/sua/AVFR/AVFR/first-order-model/face_landmarker.task'),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1  
        )
        self.landmarker = face_landmarker.FaceLandmarker.create_from_options(landmarker_options)

    def __len__(self):
        return len(self.videos)

    def generate_mel_spectrogram(self, audio_window, sr=16000, n_fft=800, hop_length=200, n_mels=80, time_frames=16):
        spectrogram = librosa.stft(audio_window, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(S=np.abs(spectrogram), sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < time_frames:
            pad_width = time_frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0,0),(0, pad_width)), mode='constant',
                                constant_values=mel_spec_db.min())
        else:
            mel_spec_db = mel_spec_db[:, :time_frames]
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
        return mel_spec_db

    def calculate_center_scale(self, image_numpy):
        image_uint8 = (image_numpy * 255).astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
        detection_result = self.landmarker.detect(mp_image)

        h, w, _ = image_numpy.shape
        if not detection_result.face_landmarks:
            return None, None, False
        
        landmarks = detection_result.face_landmarks[0]
        x_coords = [landmark.x * w for landmark in landmarks]
        y_coords = [landmark.y * h for landmark in landmarks]
        
        if not x_coords or not y_coords:
            return None, None, False

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32)
        scale = max(x_max - x_min, y_max - y_min)

        if scale < 1.0:
            print(f"Warning: Detected face scale is too small ({scale:.4f}). Treating as failure.")
            return None, None, False

        scale *= 1.2
        
        return center, scale, True

    def __getitem__(self, idx):
        if self.is_train:
            while True:
                path = self.videos[idx]
                video_name = os.path.basename(path)
                try:
                    video_array = read_video(path, frame_shape=self.frame_shape)
                except Exception as e:     
                    print(f"Video reading failed for {path}, sampling another. Error: {e}")
                    idx = np.random.randint(0, len(self.videos))
                    continue

                if len(video_array) < 5: 
                    idx = np.random.randint(0, len(self.videos))
                    continue
                
                num_frames = len(video_array)
                source_idx = np.random.randint(0, num_frames)
                driving_idx = np.random.randint(2, num_frames - 2)

                source_video_array = video_array[source_idx]
                driving_video_array = video_array[driving_idx]

                source_center, source_scale, source_ok = self.calculate_center_scale(source_video_array)
                driving_center, driving_scale, driving_ok = self.calculate_center_scale(driving_video_array)
             
                if source_ok and driving_ok:
                    driving_indices_for_audio = [driving_idx]
                    break 
                else:
                    print(f"Face detection failed for {path}, retrying with another video.")
                    idx = np.random.randint(0, len(self.videos))
                    
        else: 
            path = self.videos[idx]
            video_name = os.path.basename(path)
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            driving_video_array = [video_array[i] for i in range(num_frames)]
            driving_indices_for_audio = range(num_frames)
            source_video_idx = np.random.randint(0, len(self.videos))
            while source_video_idx == idx: 
                source_video_idx = np.random.randint(0, len(self.videos))
            source_path = self.videos[source_video_idx]
            source_full_video_array = read_video(source_path, frame_shape=self.frame_shape)
            source_frame_idx = np.random.randint(0, len(source_full_video_array))
            source_video_array = source_full_video_array[source_frame_idx]
            source_center, source_scale, _ = self.calculate_center_scale(source_video_array)
            driving_center, driving_scale, _ = self.calculate_center_scale(driving_video_array[0])

        current_mode = 'train' if self.is_train else 'test'
        base_root_dir = os.path.dirname(self.root_dir) if self.root_dir.endswith(('train', 'test')) else self.root_dir
        rel_path = os.path.relpath(path, os.path.join(base_root_dir, current_mode))
        audio_path = os.path.join(self.audio_dir, current_mode, rel_path).replace(".mp4", ".mp3")
        
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Audio loading failed for {audio_path}, returning empty audio.")
            audio = np.zeros(16000 * 5) 
            sr = 16000
        
        frame_duration = 1 / 25
        window_length = int(0.2 * sr)
        audio_inputs = []
        for frame_idx in driving_indices_for_audio:
            start_sample = int(frame_idx * frame_duration * sr) - window_length // 2
            start_sample = max(0, min(start_sample, len(audio) - window_length))
            audio_window = audio[start_sample: start_sample + window_length]
            mel_spectrogram = self.generate_mel_spectrogram(audio_window)
            audio_inputs.append(torch.tensor(mel_spectrogram, dtype=torch.float32))

        if self.transform is not None:
            source_video_array = self.transform([source_video_array])[0]
            if self.is_train:
                driving_video_array = self.transform([driving_video_array])[0]
        
        out = {}
        if self.is_train:
            source = np.array(source_video_array, dtype='float32')
            driving = np.array(driving_video_array, dtype='float32')
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['audio'] = audio_inputs[0]
        else:
            source = np.array(source_video_array, dtype='float32')
            video = np.array(driving_video_array, dtype='float32')
            out['source'] = source.transpose((2, 0, 1))
            out['video'] = video.transpose((0, 3, 1, 2))
            out['audio'] = audio_inputs

        out['name'] = video_name
        out['source_center'] = source_center
        out['source_scale'] = source_scale
        out['driving_center'] = driving_center
        out['driving_scale'] = driving_scale
        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
