import random

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from ..image_scale import check_exact_scale, rescale_intrinsics, snapped_size


class RGBDDataset(data.Dataset):
    def __init__(self, datapath, n_frames=15, train_scale=0.5):
        """ Base class for RGBD dataset """
        self.root = datapath
        self.n_frames = n_frames
        self.train_scale = train_scale

        self.scene_info = self._load_index()
        self._build_dataset_index()

    def _build_dataset_index(self):
        self.dataset_index = []
        train, held_out = [], []
        for scene, info in self.scene_info.items():
            if self.is_test_scene(scene):
                held_out.append(scene)
                continue
            train.append(scene)
            # clip start frames: the last 65 frames of a trajectory never start a clip
            for i in range(len(info['images']) - 65):
                self.dataset_index.append((scene, i))

        if len(self.dataset_index) == 0:
            raise RuntimeError(f"No training clips in the dataset index (datapath='{self.root}')")
        # trajectory keys are '<scene>/<difficulty>/<trajectory>'
        train_scenes = {key.split('/')[0] for key in train}
        print(f"Dataset: {len(train)} trajectories of {len(train_scenes)} scenes for training, "
              f"{len(held_out)} evaluation trajectories held out, "
              f"{len(self.dataset_index)} training clips")

    def __getitem__(self, index):
        """ return training video """
        # a clip with an unreadable frame is replaced by a random clip (at most 10 tries)
        for attempt in range(10):
            try:
                return self._load_clip(index)
            except OSError:
                if attempt == 9:
                    raise
                index = random.randint(0, len(self.dataset_index) - 1)

    def _load_clip(self, index):
        scene_id, ix = self.dataset_index[index]
        info = self.scene_info[scene_id]

        images, depths, poses, intrinsics = [], [], [], []
        for i in range(ix, ix + self.n_frames):
            images.append(self.image_read(info['images'][i]))
            depths.append(self.depth_read(info['depths'][i]))
            poses.append(info['poses'][i])
            intrinsics.append(info['intrinsics'][i])

        images = np.stack(images).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)
        assert images.shape[1:3] == depths.shape[1:] == self.EQUIRECT_SIZE, scene_id

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)

        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        if self.train_scale != 1.0:
            images, disps, intrinsics = self._resize_training(
                images, disps, intrinsics, self.train_scale)

        # normalize depth
        s = self._depth_normalization_scale(disps)
        disps = disps / s
        poses[...,:3] *= s

        return images, poses, disps, intrinsics

    @staticmethod
    def _depth_normalization_scale(disps, quantile=0.98, factor=0.7, max_samples=262144):
        # torch.quantile has an input size limit: use a 1/4-stride subsample, capped at max_samples
        sample = disps[:, 1::4, 1::4].reshape(-1)
        n = sample.numel()
        if n > max_samples:
            idx = torch.linspace(0, n - 1, max_samples).long()
            sample = sample[idx]
        return factor * torch.quantile(sample, quantile)

    @staticmethod
    def _resize_training(images, disps, intrinsics, scale):
        _, _, ht, wd = images.shape
        ht1, wd1 = snapped_size(ht, wd, scale)
        check_exact_scale(ht, wd, ht1, wd1, scale)

        images = F.interpolate(images, (ht1, wd1), mode='bicubic', align_corners=False)
        # nearest: no inverse-depth averaging across depth discontinuities
        disps = F.interpolate(disps.unsqueeze(1), (ht1, wd1), mode='nearest').squeeze(1)
        intrinsics = rescale_intrinsics(intrinsics, wd1 / wd, ht1 / ht)
        return images, disps, intrinsics

    def __len__(self):
        return len(self.dataset_index)
