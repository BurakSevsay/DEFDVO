"""Spherically initialized, latitude-conditioned DCNv2 encoder for equirectangular images."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .spherenet.GridGenerator import GridGenerator

DIM = 32

# azimuth padding of the input image (pixels, must be divisible by 4)
SEAM_PAD = 32


def latitude_maps(h, device, dtype):
    """[1, 2, h, 1] with sin(lat), cos(lat); rows span the full latitude range."""
    lat = ((torch.arange(h, device=device, dtype=dtype) + 0.5) / h - 0.5) * math.pi
    return torch.stack([torch.sin(lat), torch.cos(lat)])[None, :, :, None]


def sphere_base_offset(h, w, stride, max_dx):
    """SphereNet 3x3 sampling offsets [1, 18, h_out, 1] for an unpadded h x w panorama."""
    grid = GridGenerator(h, w, (3, 3), (stride, stride)).createSamplingPattern()
    h_out = int(np.ceil(h / stride))
    w_out = int(np.ceil(w / stride))
    grid = grid.reshape(h_out, 3, w_out, 3, 2)  # (lat_px, lon_px)

    # the pattern depends only on the row, so take column 0
    lat_px = grid[:, :, 0, :, 0]  # [h_out, 3, 3]
    lon_px = grid[:, :, 0, :, 1]

    ky = np.arange(3) - 1
    y_reg = (np.arange(h_out) * stride)[:, None, None] + ky[None, :, None]
    x_reg = ky[None, None, :]  # column 0

    dy = lat_px - y_reg
    dx = lon_px - x_reg
    dx = (dx + w / 2) % w - w / 2  # shortest way around the azimuth circle

    # keep taps inside the circular pad (binds only near the poles)
    dx = np.clip(dx, -max_dx, max_dx)

    base = np.stack([dy, dx], axis=-1)                # [h_out, 3, 3, 2]
    base = base.reshape(h_out, 18).T                  # k row-major, (dy, dx)
    return torch.from_numpy(base.astype(np.float32)).view(1, 18, h_out, 1)


class LatDeformConv2d(nn.Module):
    """3x3 modulated deformable conv; offsets = SphereNet prior + offset_net([x, sin(lat), cos(lat)])."""

    def __init__(self, in_planes, planes, pad_px, stride=1):
        super().__init__()
        self.stride = stride
        self.pad_px = pad_px  # circular padding of the input at this layer's scale

        self.weight = nn.Parameter(torch.empty(planes, in_planes, 3, 3))
        self.bias = nn.Parameter(torch.zeros(planes))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        # inputs: features + sin/cos latitude; outputs: 18 offset + 9 modulation channels
        self.offset_net = nn.Conv2d(in_planes + 2, 27, 3, stride, 1)
        nn.init.zeros_(self.offset_net.weight)
        nn.init.zeros_(self.offset_net.bias)

        self._cache_key = None
        self._lat = None
        self._base = None

    def _update_cache(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if self._cache_key == key:
            return
        self._cache_key = key
        self._lat = latitude_maps(h, device, dtype)
        w_true = w - 2 * self.pad_px
        self._base = sphere_base_offset(h, w_true, self.stride, max_dx=self.pad_px) \
            .to(device=device, dtype=dtype)

    def forward(self, x):
        b, _, h, w = x.shape
        self._update_cache(h, w, x.device, x.dtype)

        inp = torch.cat([x, self._lat.expand(b, 2, h, w)], dim=1)

        out = self.offset_net(inp)
        offset = out[:, :18] + self._base
        # modulation in (0, 2): unit gain at the zero init
        mask = 2 * torch.sigmoid(out[:, 18:])

        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=(self.stride, self.stride),
                             padding=(1, 1), mask=mask)


class DCNResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn, pad_px, stride=1):
        super().__init__()

        self.conv1 = LatDeformConv2d(in_planes, planes, stride=stride, pad_px=pad_px)
        self.conv2 = LatDeformConv2d(planes, planes, stride=1, pad_px=pad_px // stride)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(1, 1), stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class DCNEncoder(nn.Module):
    """Stride-4 encoder with deformable residual blocks and circular azimuth padding."""

    def __init__(self, output_dim=128, norm_fn='instance'):
        super().__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # RGB + sin/cos latitude
        self.conv1 = nn.Conv2d(3 + 2, DIM, kernel_size=(7, 7), stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM, stride=1, pad_px=SEAM_PAD // 2)
        self.layer2 = self._make_layer(2 * DIM, stride=2, pad_px=SEAM_PAD // 2)

        self.conv2 = nn.Conv2d(2 * DIM, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # zero-init offset predictors (the loop above also re-initialized them)
        for m in self.modules():
            if isinstance(m, LatDeformConv2d):
                nn.init.zeros_(m.offset_net.weight)
                nn.init.zeros_(m.offset_net.bias)

    def _make_layer(self, dim, pad_px, stride=1):
        layer1 = DCNResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride, pad_px=pad_px)
        layer2 = DCNResidualBlock(dim, dim, self.norm_fn, stride=1, pad_px=pad_px // stride)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b * n, c1, h1, w1)

        lat = latitude_maps(h1, x.device, x.dtype)
        x = torch.cat([x, lat.expand(b * n, 2, h1, w1)], dim=1)

        x = F.pad(x, (SEAM_PAD, SEAM_PAD, 0, 0), mode='circular')

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        # crop the padding at 1/4 resolution
        p = SEAM_PAD // 4
        x = x[..., p:-p]

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
