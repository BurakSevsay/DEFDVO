import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dpvo.data_readers.tartan import TartanAir
from dpvo.image_scale import equirect_scale_arg
from dpvo.lietorch import SE3
from dpvo.logger import Logger
from dpvo.net import VONet

CHECKPOINT_FREQ = 10000


def save_checkpoint(net, args, total_steps, tag=None):
    os.makedirs('checkpoints', exist_ok=True)
    suffix = tag if tag is not None else '%06d' % total_steps
    path = 'checkpoints/%s_%s.pth' % (args.name, suffix)
    payload = {
        'model': net.state_dict(),
        'steps': total_steps,
    }
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(net, args):
    payload = torch.load(args.ckpt, weights_only=True)
    net.load_state_dict(payload['model'], strict=True)


def kabsch_umeyama(A, B):
    n = A.shape[0]
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    _, D, _ = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def train(args):
    """ main training loop """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    db = TartanAir(datapath=args.datapath, n_frames=args.n_frames, train_scale=args.train_scale)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=args.workers)

    net = VONet()
    net.cuda()

    if args.ckpt is not None:
        load_checkpoint(net, args)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.steps,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy='linear',
    )

    logger = Logger(args.name, scheduler)

    total_steps = 0

    while 1:
        for data_blob in train_loader:
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # fix poses to gt for first 1k steps
            so = total_steps < 1000 and args.ckpt is None

            poses = SE3(poses).inv()
            traj = net(images, poses, disps, intrinsics, STEPS=18, structure_only=so)

            # azimuth period of the 1/4-res grid the flow coords live on
            wd_q = images.shape[-1] // 4

            loss = 0.0
            for i, (v, x, y, P1, P2) in enumerate(traj):
                # horizontal error wrapped across the azimuth seam
                dx = x[..., 0] - y[..., 0]
                dx = torch.remainder(dx + wd_q / 2, wd_q) - wd_q / 2
                dy = x[..., 1] - y[..., 1]
                # eps: sqrt has an unbounded gradient at exactly zero error
                e = torch.sqrt(dx**2 + dy**2 + 1e-12)
                e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=1).values

                N = P1.shape[1]
                ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                ii = ii.reshape(-1).cuda()
                jj = jj.reshape(-1).cuda()

                k = ii != jj
                ii = ii[k]
                jj = jj[k]

                P1 = P1.inv()
                P2 = P2.inv()

                t1 = P1.matrix()[...,:3,3]
                t2 = P2.matrix()[...,:3,3]

                s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                P1 = P1.scale(s.view(1, 1))

                dP = P1[:,ii].inv() * P1[:,jj]
                dG = P2[:,ii].inv() * P2[:,jj]

                e1 = (dP * dG.inv()).log()
                tr = e1[...,0:3].norm(dim=-1)
                ro = e1[...,3:6].norm(dim=-1)

                loss += args.flow_weight * e.mean()
                if not so and i >= 2:
                    loss += args.pose_weight * (tr.mean() + ro.mean())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            metrics = {
                "loss": loss.item(),
                "px1": (e < .25).float().mean().item(),
                "ro": ro.float().mean().item(),
                "tr": tr.float().mean().item(),
                "r1": (ro < .001).float().mean().item(),
                "r2": (ro < .01).float().mean().item(),
                "t1": (tr < .001).float().mean().item(),
                "t2": (tr < .01).float().mean().item(),
            }

            logger.push(metrics)

            if total_steps % CHECKPOINT_FREQ == 0:
                save_checkpoint(net, args, total_steps)

            if total_steps >= args.steps:
                if total_steps % CHECKPOINT_FREQ != 0:
                    save_checkpoint(net, args, total_steps, tag='final')
                logger.close()
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='defdvo', help='name of the run')
    parser.add_argument('--ckpt', help='checkpoint to initialize from (skips the structure-only phase)')
    parser.add_argument('--datapath', default='./tartanair2', help='TartanAirV2 root')
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--train_scale', type=equirect_scale_arg, default=0.5,
                        help='image scale, a multiple of 1/64 in (0, 1]')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    print("--- Experiment Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("--------------------------------")

    train(args)
