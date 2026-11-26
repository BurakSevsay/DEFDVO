#!/usr/bin/env python3
"""Build the TartanAirV2 equirect training index read by dpvo.data_readers.tartan.TartanAir."""
import argparse
import glob
import os
import os.path as osp
import pickle

from dpvo.data_readers.tartan import EQUIRECT_INDEX_FILE, TartanAir

TRAJECTORIES = ('Data_easy/P000', 'Data_hard/P000', 'Data_hard/P001')  # indexed in every scene
IMAGE_DIR = 'image_lcam_equirect'
DEPTH_DIR = 'depth_lcam_equirect'
POSE_FILE = 'pose_lcam_front.txt'


def build_index(root):
    traj_dirs = []
    for scene_dir in sorted(p for p in glob.glob(osp.join(root, '*')) if osp.isdir(p)):
        for traj in TRAJECTORIES:
            traj_dir = osp.join(scene_dir, *traj.split('/'))
            if (osp.isdir(osp.join(traj_dir, IMAGE_DIR)) and osp.isdir(osp.join(traj_dir, DEPTH_DIR))
                    and osp.isfile(osp.join(traj_dir, POSE_FILE))):
                traj_dirs.append(traj_dir)
            else:
                print(f"[missing] {osp.basename(scene_dir)}/{traj} "
                      f"(expected {IMAGE_DIR}/, {DEPTH_DIR}/, {POSE_FILE})")

    index = {}
    for traj_dir in sorted(traj_dirs):
        key = osp.relpath(traj_dir, root).replace(os.sep, '/')
        images = sorted(glob.glob(osp.join(traj_dir, IMAGE_DIR, '*.png')))
        depths = sorted(glob.glob(osp.join(traj_dir, DEPTH_DIR, '*.png')))
        if len(images) == 0 or len(images) != len(depths):
            print(f"[skip] {key}: {len(images)} images, {len(depths)} depths")
            continue

        poses = TartanAir.pose_read(osp.join(traj_dir, POSE_FILE))
        if len(poses) < len(images):
            print(f"[skip] {key}: {len(poses)} poses, {len(images)} images")
            continue
        poses = poses[:len(images)]
        poses[:, :3] /= TartanAir.DEPTH_SCALE

        index[key] = {
            'images': [osp.relpath(p, root).replace(os.sep, '/') for p in images],
            'depths': [osp.relpath(p, root).replace(os.sep, '/') for p in depths],
            'poses': poses,
        }
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='TartanAirV2 root')
    parser.add_argument('--output', default=EQUIRECT_INDEX_FILE, help='output pickle')
    args = parser.parse_args()

    index = build_index(args.root)
    if not index:
        raise RuntimeError(f"No complete trajectories found under '{args.root}'")

    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_frames = sum(len(info['images']) for info in index.values())
    print(f"Saved {len(index)} trajectories ({n_frames} frames) to {args.output}")


if __name__ == '__main__':
    main()
