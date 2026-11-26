import math

import torch

MIN_DEPTH = 0.2
EPS = 1e-6

def x_period(fx):
    """ horizontal image period in pixels (fx = W / 2pi at every scale) """
    return 2.0 * math.pi * fx

def wrap_dx(dx, fx):
    """ shortest signed horizontal distance on the azimuth circle """
    W = x_period(fx)
    return torch.remainder(dx + 0.5 * W, W) - 0.5 * W

def flow_norm(coords1, coords0, fx):
    """ flow magnitude, x component wrapped across the seam """
    dx = coords1[...,0] - coords0[...,0]
    dx = wrap_dx(dx, fx)
    dy = coords1[...,1] - coords0[...,1]
    return torch.sqrt(dx**2 + dy**2)


def iproj(patches, intrinsics):
    """ inverse projection """
    x, y, d = patches.unbind(dim=2)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    theta = (x - cx) / fx
    phi = (y - cy) / fy

    ctheta = torch.cos(theta)
    stheta = torch.sin(theta)
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)

    xn = stheta * cphi
    yn = sphi
    zn = ctheta * cphi

    X = torch.stack([xn, yn, zn, d], dim=-1)
    return X


def proj(X, intrinsics):
    """ projection """

    X, Y, Z, _ = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    rho = torch.sqrt((X * X + Z * Z).clamp(min=EPS))
    theta = torch.atan2(X, Z)
    phi = torch.atan2(Y, rho)

    # wrap azimuth into [0, image width)
    x = torch.remainder(fx * theta + cx, x_period(fx))
    y = fy * phi + cy

    return torch.stack([x, y], dim=-1)


def transform(poses, patches, intrinsics, ii, jj, kk, jacobian=False, tonly=False):
    """ projective transform """

    # backproject
    X0 = iproj(patches[:,kk], intrinsics[:,ii])

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    if tonly:
        Gij[...,3:] = torch.as_tensor([0,0,0,1], device=Gij.device)

    X1 = Gij[:,:,None,None] * X0

    # project
    x1 = proj(X1, intrinsics[:,jj])


    if jacobian:
        p = X1.shape[2]
        X, Y, Z, H = X1[...,p//2,p//2,:].unbind(dim=-1)
        o = torch.zeros_like(H)

        fx, fy, _, _ = intrinsics[:,jj].unbind(dim=-1)

        rho2 = (X * X + Z * Z).clamp(min=EPS)
        rho = torch.sqrt(rho2)
        s2 = (X * X + Y * Y + Z * Z).clamp(min=EPS)

        du_dX = fx * Z / rho2
        du_dY = o
        du_dZ = -fx * X / rho2

        dv_dX = -fy * X * Y / (rho * s2)
        dv_dY = fy * rho / s2
        dv_dZ = -fy * Z * Y / (rho * s2)

        Ja = torch.stack([
            H,  o,  o,  o,  Z, -Y,
            o,  H,  o, -Z,  o,  X,
            o,  o,  H,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(1, len(ii), 4, 6)

        Jp = torch.stack([
            du_dX, du_dY, du_dZ, o,
            dv_dX, dv_dY, dv_dZ, o,
        ], dim=-1).view(1, len(ii), 2, 4)

        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None].adjT(Jj)

        Jz = torch.matmul(Jp, Gij.matrix()[...,:,3:])

        # minimum range
        valid_mask = ((X * X + Y * Y + Z * Z) > (MIN_DEPTH * H).pow(2)).float()
        return x1, valid_mask, (Ji, Jj, Jz)

    return x1


def flow_mag(poses, patches, intrinsics, ii, jj, kk, beta=0.3):
    """ weighted full / translation-only flow magnitude """

    coords0 = transform(poses, patches, intrinsics, ii, ii, kk)
    coords1 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False)
    coords2 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    fx = intrinsics[:, jj, 0, None, None]
    flow1 = flow_norm(coords1, coords0, fx)
    flow2 = flow_norm(coords2, coords0, fx)

    return beta * flow1 + (1-beta) * flow2
