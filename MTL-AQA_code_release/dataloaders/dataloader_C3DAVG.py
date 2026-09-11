# Fixed dataloader_C3DAVG.py - Handles sequential frame mapping correctly
# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Fixed by: AI Assistant (frame mapping bug fix)

import json
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
from opts import *

torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)
random.seed(randomseed)
np.random.seed(randomseed)
torch.backends.cudnn.deterministic = True

def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

class VideoDataset(Dataset):
    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def __init__(self, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode

        # Load annotations
        self.annotations = pkl.load(open(os.path.join(anno_n_splits_dir, 'final_annotations_dict.pkl'), 'rb'))

        if self.mode == 'train':
            self.keys = pkl.load(open(os.path.join(anno_n_splits_dir, 'train_split_' + str(randomseed) + '.pkl'), 'rb'))
        elif self.mode == 'test':
            self.keys = pkl.load(open(os.path.join(anno_n_splits_dir, 'test_split_' + str(randomseed) + '.pkl'), 'rb'))

        # Load captions if needed
        if with_caption:
            self.captions = pkl.load(open(os.path.join(anno_n_splits_dir, 'final_captions_dict.pkl'), 'rb'))
            info = json.load(open(os.path.join(anno_n_splits_dir, 'vocab.json'), 'r'))
            self.ix_to_word = info['ix_to_word']
            self.word_to_ix = info['word_to_ix']
            self.max_cap_len = max_cap_len

        # CREATE FRAME MAPPING FOR EACH VIDEO
        # This recreates the same mapping the extraction script used
        print(f"Creating frame mapping for {mode} set...")
        self.frame_mappings = {}
        self._create_frame_mappings()
        print(f"Frame mappings created for {len(self.frame_mappings)} videos")

    def _create_frame_mappings(self):
        """Create mapping from original frame numbers to sequential indices for each video"""
        # Get all keys (train + test) to calculate complete mapping
        all_keys_path = os.path.join(anno_n_splits_dir, 'train_split_' + str(randomseed) + '.pkl')
        test_keys_path = os.path.join(anno_n_splits_dir, 'test_split_' + str(randomseed) + '.pkl')

        train_keys = pkl.load(open(all_keys_path, 'rb'))
        test_keys = pkl.load(open(test_keys_path, 'rb'))
        all_keys = train_keys + test_keys

        # Group by video and collect all needed frames
        video_frames = {}
        for key in all_keys:
            video_id = key[0]
            anno = self.annotations.get(key)

            if anno and anno.get('end_frame'):
                end_frame = anno['end_frame']
                # Include temporal augmentation range
                start_idx = max(0, end_frame + temporal_aug_min - sample_length)
                end_idx = end_frame + temporal_aug_max

                if video_id not in video_frames:
                    video_frames[video_id] = set()

                video_frames[video_id].update(range(start_idx, end_idx + 1))

        # Create sorted mapping for each video
        for video_id, frames_set in video_frames.items():
            frames_list = sorted(frames_set)
            # Map: original_frame_number -> sequential_index
            frame_to_sequential = {frame_num: idx for idx, frame_num in enumerate(frames_list)}
            self.frame_mappings[video_id] = frame_to_sequential

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ix):
        transform = transforms.Compose([
            transforms.CenterCrop(H),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Get video info
        video_id = self.keys[ix][0]
        video_dir = os.path.join(dataset_frames_dir, str('{:02d}'.format(video_id)))

        # Get frame mapping for this video
        frame_mapping = self.frame_mappings.get(video_id, {})

        # Get end frame from annotations
        end_frame = self.annotations.get(self.keys[ix]).get('end_frame')

        # Temporal augmentation
        if self.mode == 'train':
            temporal_aug_shift = random.randint(temporal_aug_min, temporal_aug_max)
            end_frame = end_frame + temporal_aug_shift

        start_frame = end_frame - sample_length

        # Spatial augmentation
        if self.mode == 'train':
            hori_flip = random.randint(0, 1)

        # Initialize image tensor
        images = torch.zeros(sample_length, C, H, W)

        # Load frames using correct sequential indices
        for i in range(sample_length):
            original_frame = start_frame + i

            # Get sequential index for this original frame
            if original_frame in frame_mapping:
                sequential_idx = frame_mapping[original_frame]
                frame_path = os.path.join(video_dir, f'{sequential_idx:06d}.jpg')

                if os.path.exists(frame_path):
                    if self.mode == 'train':
                        images[i] = load_image_train(frame_path, hori_flip, transform)
                    else:
                        images[i] = load_image(frame_path, transform)

        # Get labels
        label_final_score = self.annotations.get(self.keys[ix]).get('final_score')
        label_position = self.annotations.get(self.keys[ix]).get('position')
        label_armstand = self.annotations.get(self.keys[ix]).get('armstand')
        label_rot_type = self.annotations.get(self.keys[ix]).get('rotation_type')
        label_ss_no = self.annotations.get(self.keys[ix]).get('ss_no')
        label_tw_no = self.annotations.get(self.keys[ix]).get('tw_no')

        # Process captions if in training mode
        if self.mode == 'train' and with_caption:
            label_captions = np.zeros(self.max_cap_len)
            label_captions_mask = np.zeros(self.max_cap_len)
            captions = self.captions.get(self.keys[ix])

            if captions is None:
                print(f'Warning: No captions for {self.keys[ix]}')
                captions = []

            # Truncate if too long
            if len(captions) > self.max_cap_len:
                captions = captions[:self.max_cap_len]

            # Process captions, skip empty strings
            for j, w in enumerate(captions):
                if w and w in self.word_to_ix:
                    label_captions[j] = self.word_to_ix[w]
                elif w:
                    # Use unknown token
                    if '<UNK>' in self.word_to_ix:
                        label_captions[j] = self.word_to_ix['<UNK>']
                    else:
                        label_captions[j] = 0

            # Create mask
            label_captions_non_zero = (label_captions == 0).nonzero()
            if len(label_captions_non_zero[0]) > 0:
                label_captions_mask[:int(label_captions_non_zero[0][0]) + 1] = 1
            else:
                label_captions_mask[:] = 1

        # Prepare output data
        data = {}
        data['video'] = images
        data['label_position'] = label_position
        data['label_armstand'] = label_armstand
        data['label_rot_type'] = label_rot_type
        data['label_ss_no'] = label_ss_no
        data['label_tw_no'] = label_tw_no
        data['label_final_score'] = label_final_score / final_score_std

        if self.mode == 'train' and with_caption:
            data['label_captions'] = torch.from_numpy(label_captions).type(torch.LongTensor)
            data['label_captions_mask'] = torch.from_numpy(label_captions_mask).type(torch.FloatTensor)

        return data
