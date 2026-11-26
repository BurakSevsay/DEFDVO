import argparse
import datetime
import glob
import os
import os.path as osp
import re
import traceback

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation

from dpvo.config import cfg
from dpvo.data_readers.tartan import equirect_intrinsics
from dpvo.dpvo import DPVO
from dpvo.image_scale import resize_tensor, snapped_size, valid_equirect_scale
from dpvo.metrics import BAD_SCORE, ate_rmse, is_bad, rpe
from dpvo.plot_utils import plot_trajectory

MAX_FAILED_TRIALS = 2
NATIVE_HT, NATIVE_WD = 1920, 3840

# Dataset card order, i.e. Easy 00-09 and Hard 00-09 ("shanghai driving" on the card is shanghai_street).
EASY_SEQUENCES = ["bridge_night", "canyon_line", "city_driving", "downhill_biking", "hongkong_central",
                  "hongkong_wanchai", "mountains", "shanghai_street", "snowmobile", "tokyo_citywalk"]
HARD_SEQUENCES = ["canyon_loop", "dragon_boat", "drone_racetrack", "field", "grove",
                  "indoor_RC_car", "london_bridge", "ridge_to_lake", "snowy_mountain_road", "wingsuit"]

# Ry(-90 deg), xyzw: maps DEFDVO's body frame (forward at column 3W/4, the cx of equirect_intrinsics)
# to the 360DVO camera frame (forward at column W/2).
BODY_FIX_XYZW = (0.0, -0.7071067811865476, 0.0, 0.7071067811865476)


def eval_scale_arg(value):
    scale = float(value)
    if not valid_equirect_scale(scale, ht=NATIVE_HT, wd=NATIVE_WD):
        raise argparse.ArgumentTypeError(
            f"scale {scale}: {NATIVE_WD}x{NATIVE_HT} * scale must be integers divisible by 16")
    return scale


def sequence_paths(datapath, sequence):
    return osp.join(datapath, "Sequences", sequence), osp.join(datapath, "GroundTruth", f"{sequence}.txt")


def load_gt(gt_file):
    """Camera-to-world (N, 7) [tx ty tz qx qy qz qw] poses, one row per image."""
    poses = np.loadtxt(gt_file, ndmin=2)
    if poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError(f"Expected finite rows 'tx ty tz qx qy qz qw' in {gt_file}")
    # world-to-camera (x right, y up, z back) -> camera-to-world (x right, y down, z forward)
    poses[:, 3:7] = poses[:, [6, 5, 4, 3]] * np.array([1.0, -1.0, 1.0, 1.0])
    return poses


def body_frame_fix(poses):
    """Right-multiply the orientations of (N, 7) poses by BODY_FIX_XYZW; positions are unchanged."""
    poses = np.array(poses, dtype=np.float64, copy=True)
    poses[:, 3:7] = (Rotation.from_quat(poses[:, 3:7]) * Rotation.from_quat(BODY_FIX_XYZW)).as_quat()
    return poses


def video_iterator(imagedir, eval_scale=0.5):
    imfiles = sorted(glob.glob(osp.join(imagedir, "*.jpg")))
    base_intrinsics = torch.as_tensor(equirect_intrinsics(*cv2.imread(imfiles[0]).shape[:2]))

    for imfile in imfiles:
        image = torch.from_numpy(cv2.imread(imfile)).permute(2, 0, 1)
        image, intrinsics = resize_tensor(image, base_intrinsics, eval_scale)
        yield image.cuda(), intrinsics.cuda()


@torch.no_grad()
def run(imagedir, cfg, network, eval_scale=0.5):
    imfiles = sorted(glob.glob(osp.join(imagedir, "*.jpg")))
    if not imfiles:
        raise FileNotFoundError(f"No JPG images found in {imagedir}")

    ht, wd = snapped_size(*cv2.imread(imfiles[0]).shape[:2], eval_scale)
    slam = DPVO(cfg, network, ht=ht, wd=wd)

    for t, (image, intrinsics) in enumerate(video_iterator(imagedir, eval_scale)):
        slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    return slam.terminate()


def to_evo(poses, tstamps):
    """PoseTrajectory3D from (N, 7) [x y z qx qy qz qw] poses."""
    return PoseTrajectory3D(positions_xyz=poses[:, :3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
                            timestamps=tstamps)


def median(values):
    values = [v for v in values if not is_bad(v)]
    return float(np.median(values)) if values else None


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def fmt(scores):
    return " | ".join(f"{k} {v:.9f}" if v is not None else f"{k} N/A"
                      for k, v in zip(("ATE", "RPE-t", "RPE-r"), scores))


def summarize(trials):
    """Median scores over the successful trials, or None if the sequence failed."""
    ok = [t for t in trials if not is_bad(t[0])]
    if not ok or len(trials) - len(ok) > MAX_FAILED_TRIALS:
        return None
    return [median(v) for v in zip(*ok)]


def write_report(path, header, results, groups):
    """Write per-trial scores, sequence medians and group means; return the group lines."""
    medians = {seq: summarize(trials) for seq, trials in results.items()}
    lines = list(header)
    for seq, trials in results.items():
        lines += ["", seq]
        for j, t in enumerate(trials, 1):
            lines.append(f"  Trial {j:02d}: " + ("failed" if is_bad(t[0]) else fmt(t)))
        n_failed = sum(is_bad(t[0]) for t in trials)
        lines.append("  Median:   " + (fmt(medians[seq]) if medians[seq] is not None
                                       else f"FAILED ({n_failed}/{len(trials)} trials failed)"))

    summary, averages = [], []
    for group, seqs in groups.items():
        seqs = [s for s in seqs if s in results]
        ok = [medians[s] for s in seqs if medians[s] is not None]
        if seqs:
            avg = [mean(v) for v in zip(*ok)] if len(ok) == len(seqs) else [None] * 3
            averages.append(avg)
            summary.append(f"{group}: {fmt(avg)} ({len(seqs)} sequences, {len(seqs) - len(ok)} failed)")
    if len(averages) == 2 and None not in averages[0] + averages[1]:
        summary.append(f"avg (easy + hard) / 2: {fmt([mean(v) for v in zip(*averages)])}")

    with open(path, "w") as f:
        f.write("\n".join(lines + [""] + summary) + "\n")
    return summary


def result_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return name or datetime.datetime.now().strftime("evaluation_%Y%m%d_%H%M%S")


def evaluate(args, sequences):
    name = result_name(args.name)
    os.makedirs(args.results_dir, exist_ok=True)
    report = osp.join(args.results_dir, f"{name}.txt")
    header = [f"360DVO: weights {args.weights}, config {' '.join([args.config] + args.opts)}, "
              f"eval_scale {args.eval_scale}, {args.trials} trials, seed {args.seed}",
              "ATE [m] | RPE-t [m/frame] | RPE-r [deg/frame] after Sim(3) alignment. Sequence: median over successful "
              f"trials, failed if more than {MAX_FAILED_TRIALS} trials fail. Splits: mean of the sequence medians, "
              "N/A if any sequence failed."]
    groups = {"easy": EASY_SEQUENCES, "hard": HARD_SEQUENCES}

    results = {}
    for sequence in sequences:
        imagedir, gt_file = sequence_paths(args.datapath, sequence)
        results[sequence] = []
        for j in range(args.trials):
            try:
                traj_est, tstamps = run(imagedir, cfg, args.weights, args.eval_scale)
            except Exception:
                traceback.print_exc()
                print(f"WARNING: trial {j + 1} on {sequence} raised, counted as failed")
                results[sequence].append((BAD_SCORE,) * 3)
                continue

            traj_ref = load_gt(gt_file)
            n = min(len(traj_ref), len(traj_est))
            # body_frame_fix (scipy) cannot handle non-finite rotations
            if n < 3 or not np.isfinite(traj_est[:n, 3:]).all():
                print(f"WARNING: trial {j + 1} on {sequence} returned an invalid trajectory, counted as failed")
                results[sequence].append((BAD_SCORE,) * 3)
                continue

            traj_est = to_evo(body_frame_fix(traj_est[:n]), tstamps[:n])
            traj_ref = to_evo(traj_ref[:n], tstamps[:n])
            ate = ate_rmse(traj_ref, traj_est)
            results[sequence].append((ate, *rpe(traj_ref, traj_est)))

            stem = f"{name}_360DVO_{sequence}_Trial{j + 1:02d}"
            if args.plot and not is_bad(ate):
                os.makedirs("trajectory_plots", exist_ok=True)
                plot_trajectory(traj_est, traj_ref,
                                f"360DVO {sequence.replace('_', ' ')} Trial #{j + 1} (ATE: {ate:.03f})",
                                f"trajectory_plots/{stem}.pdf")
            if args.save_trajectory:
                os.makedirs("saved_trajectories", exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/{stem}.txt", traj_est)

        print(sequence, sorted(t[0] for t in results[sequence]))
        summary = write_report(report, header, results, groups)

    print("\n".join(summary))
    print(f"Results written to {report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="saved_checkpoints/DEFDVO.pth")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--datapath', default="./360DVO/Data", help="360DVO root with Sequences/ and GroundTruth/")
    parser.add_argument('--sequences', nargs='+', help="subset of the 20 sequences")
    parser.add_argument('--trials', type=int, default=5, help="runs per sequence")
    parser.add_argument('--eval_scale', type=eval_scale_arg, default=0.5, help="image scale (0.5: 1920x960)")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--name', default="", help="report name, written to <results_dir>/<name>.txt")
    parser.add_argument('--results_dir', default="saved_results")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    args = parser.parse_args()

    all_sequences = EASY_SEQUENCES + HARD_SEQUENCES
    if args.sequences:
        unknown = sorted(set(args.sequences) - set(all_sequences))
        if unknown:
            parser.error(f"--sequences not in 360DVO: {', '.join(unknown)}")
    sequences = [s for s in all_sequences if not args.sequences or s in args.sequences]
    missing = [s for s in sequences if not all(map(osp.exists, sequence_paths(args.datapath, s)))]
    if missing:
        print(f"WARNING: skipping sequences not found in {args.datapath}: {', '.join(missing)}")
        sequences = [s for s in sequences if s not in missing]
    if not sequences:
        raise FileNotFoundError(f"No 360DVO sequences found in {args.datapath}")

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(args.seed)
    evaluate(args, sequences)
