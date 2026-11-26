import argparse

import torch
import torch.nn.functional as F

# native TartanAirV2 panorama size (height, width)
EQUIRECT_SIZE_HW = (1024, 2048)


def valid_equirect_scale(scale, ht=EQUIRECT_SIZE_HW[0], wd=EQUIRECT_SIZE_HW[1], stride=16):
    """True if (ht, wd) * scale are integers divisible by stride, which keeps fx = W/(2*pi) exact."""
    if not (0.0 < scale <= 1.0):
        return False
    for size in (ht, wd):
        s = size * scale
        if abs(s - round(s)) > 1e-6 or int(round(s)) % stride != 0:
            return False
    return True


def equirect_scale_arg(value):
    """argparse type for --train_scale/--eval_scale."""
    scale = float(value)
    if not valid_equirect_scale(scale):
        raise argparse.ArgumentTypeError(
            f"scale {scale} breaks the equirect geometry: {EQUIRECT_SIZE_HW[1]}x"
            f"{EQUIRECT_SIZE_HW[0]} * scale must be integers divisible by 16. "
            f"Use a multiple of 1/64 in (0, 1], e.g. 1.0, 0.875, 0.75, 0.625, "
            f"0.5, 0.375, 0.25, 0.125.")
    return scale


def rescale_intrinsics(intrinsics, sx, sy):
    """Rescale a [..., 4] tensor of [fx, fy, cx, cy] for a resize by (sx, sy), pixel-center aware."""
    out = intrinsics.clone()
    out[..., 0] = intrinsics[..., 0] * sx
    out[..., 1] = intrinsics[..., 1] * sy
    out[..., 2] = (intrinsics[..., 2] + 0.5) * sx - 0.5
    out[..., 3] = (intrinsics[..., 3] + 0.5) * sy - 0.5
    return out


def snapped_size(ht, wd, scale, stride=16):
    """Output size after scale, snapped down to multiples of stride."""
    ht1 = max(int(ht * scale), stride)
    wd1 = max(int(wd * scale), stride)
    ht1 -= ht1 % stride
    wd1 -= wd1 % stride
    return ht1, wd1


def check_exact_scale(ht, wd, ht1, wd1, scale, stride=16):
    """Raise if the snapped size (ht1, wd1) differs from (ht, wd) * scale."""
    if (ht1, wd1) != (int(round(ht * scale)), int(round(wd * scale))):
        raise ValueError(
            f"scale {scale} on a {wd}x{ht} panorama snaps to {wd1}x{ht1} "
            f"(not {wd * scale:g}x{ht * scale:g}); this breaks fx = W/(2*pi). "
            f"Use a scale where W*s and H*s are integers divisible by {stride} "
            f"(multiples of 1/64 for 2048x1024, e.g. 1.0, 0.5, 0.375).")


def resize_tensor(image, intrinsics, scale, stride=16):
    """Resize a CHW image tensor and its intrinsics by scale."""
    _, ht, wd = image.shape
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)

    if scale == 1.0:
        if ht % stride or wd % stride:
            raise ValueError(
                f"equirect input {wd}x{ht} is not divisible by {stride}; "
                f"column-cropping a panorama would break the 2*pi azimuth period.")
        return image, intrinsics

    ht1, wd1 = snapped_size(ht, wd, scale, stride)
    check_exact_scale(ht, wd, ht1, wd1, scale, stride)
    image = F.interpolate(
        image.unsqueeze(0).float(), (ht1, wd1), mode="bicubic", align_corners=False
    ).squeeze(0)
    return image, rescale_intrinsics(intrinsics, wd1 / wd, ht1 / ht)
