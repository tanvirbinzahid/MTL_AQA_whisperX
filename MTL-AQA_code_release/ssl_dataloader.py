import os
import random
from typing import List, Tuple

from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def pil_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert('RGB')


class ClipAugment:
    def __init__(self, size: int = 112):
        self.size = size

    def _color_jitter(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.8:
            b = random.uniform(0.6, 1.4)
            c = random.uniform(0.6, 1.4)
            s = random.uniform(0.6, 1.4)
            h = random.uniform(-0.08, 0.08)
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h)
        return img

    def _blur(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        return img

    def _gray(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.2:
            img = ImageOps.grayscale(img).convert('RGB')
        return img

    def _crop_params(self, w: int, h: int) -> Tuple[int, int, int, int]:
        scale = random.uniform(0.6, 1.0)
        ratio = random.uniform(0.75, 1.3333333333)
        area = w * h * scale
        new_w = int(round((area * ratio) ** 0.5))
        new_h = int(round((area / ratio) ** 0.5))
        new_w = min(max(new_w, self.size), w)
        new_h = min(max(new_h, self.size), h)
        i = 0 if h == new_h else random.randint(0, h - new_h)
        j = 0 if w == new_w else random.randint(0, w - new_w)
        return i, j, new_h, new_w

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        w, h = frames[0].size
        i, j, th, tw = self._crop_params(w, h)
        do_flip = random.random() < 0.5
        out = []
        for img in frames:
            img = TF.resized_crop(img, i, j, th, tw, (self.size, self.size))
            if do_flip:
                img = TF.hflip(img)
            img = self._color_jitter(img)
            img = self._gray(img)
            img = self._blur(img)
            t = TF.to_tensor(img)
            t = TF.normalize(
                t,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            out.append(t)
        return torch.stack(out, dim=1)  # (C, T, H, W)


class SSLVideoClipDataset(Dataset):
    def __init__(
        self,
        frames_root: str,
        clip_len: int = 16,
        stride: int = 1,
        min_frames: int = 16,
        size: int = 112,
    ):
        self.frames_root = frames_root
        self.clip_len = clip_len
        self.stride = stride
        self.min_frames = min_frames
        self.transform = ClipAugment(size=size)
        self.video_dirs = self._discover_video_dirs()

        if not self.video_dirs:
            raise RuntimeError(f'No valid frame directories found in {frames_root}')

    def _discover_video_dirs(self) -> List[str]:
        dirs = []
        if not os.path.isdir(self.frames_root):
            return dirs
        for name in sorted(os.listdir(self.frames_root)):
            full = os.path.join(self.frames_root, name)
            if not os.path.isdir(full):
                continue
            frames = [f for f in os.listdir(full) if f.lower().endswith(IMG_EXTS)]
            if len(frames) >= self.min_frames:
                dirs.append(full)
        return dirs

    def _sample_indices(self, n_frames: int) -> List[int]:
        span = self.clip_len * self.stride
        if n_frames <= span:
            idx = list(range(0, n_frames, self.stride))
            while len(idx) < self.clip_len:
                idx.append(idx[-1])
            return idx[:self.clip_len]
        start = random.randint(0, n_frames - span)
        idx = [start + i * self.stride for i in range(self.clip_len)]
        return idx

    def _load_clip(self, video_dir: str) -> List[Image.Image]:
        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.lower().endswith(IMG_EXTS)]
        )
        idx = self._sample_indices(len(frame_files))
        return [pil_loader(os.path.join(video_dir, frame_files[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, index: int):
        video_dir = self.video_dirs[index]
        frames = self._load_clip(video_dir)
        view1 = self.transform(frames)
        view2 = self.transform(frames)
        return view1, view2, os.path.basename(video_dir)