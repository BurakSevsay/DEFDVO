import os.path as osp
import pickle

import cv2
import numpy as np

from .base import RGBDDataset

EQUIRECT_INDEX_FILE = 'tartanairv2_lcam_equirect_subset.pkl'


def equirect_intrinsics(height=1024, width=2048):
    """ERP intrinsics [fx, fy, cx, cy] (pixel centers); cx sits at 3W/4, the TartanAirV2 forward azimuth."""
    return np.array([
        width / (2.0 * np.pi),
        height / np.pi,
        0.75 * width - 0.5,
        0.5 * height - 0.5,
    ], dtype=np.float32)


# TartanAirV2 evaluation trajectories: Set1 (360DVO split) and Set2 (CubeDVO split)
SCENE_SETS = {
    'set1': [
        'House/Data_easy/P000',
        'House/Data_hard/P001',
        'VictorianStreet/Data_easy/P000',
        'VictorianStreet/Data_hard/P001',
        'OldTownNight/Data_easy/P000',
        'OldTownNight/Data_hard/P001',
        'CountryHouse/Data_easy/P000',
        'CountryHouse/Data_hard/P001',
    ],
    'set2': [
        'Antiquity3D/Data_hard/P000',
        'Antiquity3D/Data_easy/P000',
        'DesertGasStation/Data_easy/P000',
        'DesertGasStation/Data_hard/P000',
        'HongKong/Data_easy/P000',
        'HongKong/Data_hard/P000',
        'HQWesternSaloon/Data_easy/P000',
        'HQWesternSaloon/Data_hard/P000',
        'IndustrialHangar/Data_easy/P000',
        'IndustrialHangar/Data_hard/P000',
        'OldTownNight/Data_easy/P000',
        'OldTownNight/Data_hard/P000',
    ],
}

# both sets without duplicates (OldTownNight/Data_easy/P000 is in both)
test_split = list(dict.fromkeys(SCENE_SETS['set1'] + SCENE_SETS['set2']))


class TartanAir(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0
    EQUIRECT_SIZE = (1024, 2048)  # (height, width)
    EQUIRECT_INTRINSICS = equirect_intrinsics(*EQUIRECT_SIZE)

    @staticmethod
    def is_test_scene(scene):
        """True for the 19 evaluation trajectories of Set 1 and Set 2."""
        return scene in test_split

    def _load_index(self):
        for path in (EQUIRECT_INDEX_FILE, osp.join(self.root, EQUIRECT_INDEX_FILE)):
            if osp.isfile(path):
                break
        else:
            raise FileNotFoundError(
                f"{EQUIRECT_INDEX_FILE} not found in the working directory or in '{self.root}'; "
                f"build it with: python create_tartanairv2_pickle.py --root {self.root}")

        with open(path, 'rb') as f:
            scene_info = pickle.load(f)
        print(f"Loaded dataset index from {path}")

        for info in scene_info.values():
            info['images'] = [osp.join(self.root, p) for p in info['images']]
            info['depths'] = [osp.join(self.root, p) for p in info['depths']]
            # the intrinsics are fixed by the panorama size
            info['intrinsics'] = [self.EQUIRECT_INTRINSICS] * len(info['images'])
        return scene_info

    @staticmethod
    def image_read(image_file):
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to read image file: {image_file}")
        return image

    @staticmethod
    def depth_read(depth_file):
        depth_rgba = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            raise OSError(f"Failed to read depth file: {depth_file}")
        depth = depth_rgba.view("<f4")
        depth = np.squeeze(depth, axis=-1).astype(np.float32) / TartanAir.DEPTH_SCALE
        # invalid depth -> far away (disparity ~ 0)
        invalid = (~np.isfinite(depth)) | (depth <= 0.0)
        depth[invalid] = 2000.0
        return depth

    @staticmethod
    def pose_read(pose_path):
        """Read a TartanAir pose file (tx ty tz qx qy qz qw, NED) as float32 [N, 7]."""
        poses = np.loadtxt(pose_path, delimiter=' ')
        if poses.ndim == 1:
            poses = poses[None, :]
        if poses.shape[1] != 7:
            raise ValueError(
                f"Expected 7 columns (tx ty tz qx qy qz qw) in '{pose_path}', got shape {poses.shape}.")

        # NED (x forward, y right, z down) -> (x right, y down, z forward)
        return poses[:, [1, 2, 0, 4, 5, 3, 6]].astype(np.float32)
