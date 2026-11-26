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

from dpvo.config import cfg
from dpvo.data_readers.tartan import SCENE_SETS, TartanAir, test_split
from dpvo.dpvo import DPVO
from dpvo.image_scale import equirect_scale_arg, resize_tensor, snapped_size
from dpvo.metrics import BAD_SCORE, ate_rmse, is_bad, rpe
from dpvo.plot_utils import plot_trajectory

MAX_FAILED_TRIALS = 2


def scene_paths(datapath, scene):
    return osp.join(datapath, scene, "image_lcam_equirect"), osp.join(datapath, scene, "pose_lcam_front.txt")


def video_iterator(imagedir, eval_scale=0.5):
    imfiles = sorted(glob.glob(osp.join(imagedir, "*.png")))
    base_intrinsics = torch.as_tensor(TartanAir.EQUIRECT_INTRINSICS)

    for imfile in imfiles:
        image = torch.from_numpy(cv2.imread(imfile)).permute(2, 0, 1)
        image, intrinsics = resize_tensor(image, base_intrinsics, eval_scale)
        yield image.cuda(), intrinsics.cuda()


@torch.no_grad()
def run(imagedir, cfg, network, eval_scale=0.5):
    imfiles = sorted(glob.glob(osp.join(imagedir, "*.png")))
    if not imfiles:
        raise FileNotFoundError(f"No PNG images found in {imagedir}")

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
        summary.append(f"avg (set1 + set2) / 2: {fmt([mean(v) for v in zip(*averages)])}")

    with open(path, "w") as f:
        f.write("\n".join(lines + [""] + summary) + "\n")
    return summary


def result_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return name or datetime.datetime.now().strftime("evaluation_%Y%m%d_%H%M%S")


def evaluate(args, scenes):
    name = result_name(args.name)
    os.makedirs(args.results_dir, exist_ok=True)
    report = osp.join(args.results_dir, f"{name}.txt")
    header = [f"TartanAirV2 {args.split}: weights {args.weights}, config {' '.join([args.config] + args.opts)}, "
              f"eval_scale {args.eval_scale}, {args.trials} trials, seed {args.seed}",
              "ATE [m] | RPE-t [m/frame] | RPE-r [deg/frame] after Sim(3) alignment. Sequence: median over successful "
              f"trials, failed if more than {MAX_FAILED_TRIALS} trials fail. Sets: mean of the sequence medians, "
              "N/A if any sequence failed."]
    groups = {k: v for k, v in SCENE_SETS.items() if args.split in ("all", k)}

    results = {}
    for scene in scenes:
        imagedir, gt_file = scene_paths(args.datapath, scene)
        scene_name = "_".join(scene.split("/")).title()
        results[scene] = []
        for j in range(args.trials):
            try:
                traj_est, tstamps = run(imagedir, cfg, args.weights, args.eval_scale)
            except Exception:
                traceback.print_exc()
                print(f"WARNING: trial {j + 1} on {scene} raised, counted as failed")
                results[scene].append((BAD_SCORE,) * 3)
                continue

            traj_ref = TartanAir.pose_read(gt_file)
            n = min(len(traj_ref), len(traj_est))
            if n < 3:
                print(f"WARNING: trial {j + 1} on {scene} has only {n} poses, counted as failed")
                results[scene].append((BAD_SCORE,) * 3)
                continue

            traj_est = to_evo(traj_est[:n], tstamps[:n])
            traj_ref = to_evo(traj_ref[:n], tstamps[:n])
            ate = ate_rmse(traj_ref, traj_est)
            results[scene].append((ate, *rpe(traj_ref, traj_est)))

            stem = f"{name}_TartanAir_{scene_name}_Trial{j + 1:02d}"
            if args.plot and not is_bad(ate):
                os.makedirs("trajectory_plots", exist_ok=True)
                plot_trajectory(traj_est, traj_ref,
                                f"TartanAir {scene_name.replace('_', ' ')} Trial #{j + 1} (ATE: {ate:.03f})",
                                f"trajectory_plots/{stem}.pdf")
            if args.save_trajectory:
                os.makedirs("saved_trajectories", exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/{stem}.txt", traj_est)

        print(scene, sorted(t[0] for t in results[scene]))
        summary = write_report(report, header, results, groups)

    print("\n".join(summary))
    print(f"Results written to {report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="saved_checkpoints/DEFDVO.pth")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--datapath', default="./tartanair2", help="TartanAirV2 root")
    parser.add_argument('--split', default="all", choices=["all", "set1", "set2"])
    parser.add_argument('--scenes', nargs='+', help="subset of the split, e.g. House/Data_easy/P000")
    parser.add_argument('--trials', type=int, default=5, help="runs per scene")
    parser.add_argument('--eval_scale', type=equirect_scale_arg, default=0.5, help="image scale (0.5: 1024x512)")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--name', default="", help="report name, written to <results_dir>/<name>.txt")
    parser.add_argument('--results_dir', default="saved_results")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    args = parser.parse_args()

    scenes = test_split if args.split == "all" else SCENE_SETS[args.split]
    if args.scenes:
        unknown = sorted(set(args.scenes) - set(scenes))
        if unknown:
            parser.error(f"--scenes not in split {args.split}: {', '.join(unknown)}")
        scenes = [s for s in scenes if s in args.scenes]

    missing = [s for s in scenes if not all(map(osp.exists, scene_paths(args.datapath, s)))]
    if missing:
        print(f"WARNING: skipping scenes not found in {args.datapath}: {', '.join(missing)}")
        scenes = [s for s in scenes if s not in missing]
    if not scenes:
        raise FileNotFoundError(f"No evaluation scenes found in {args.datapath}")

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(args.seed)
    evaluate(args, scenes)
