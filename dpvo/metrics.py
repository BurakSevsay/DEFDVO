"""ATE and RPE after Sim(3) Umeyama alignment."""

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import numpy as np
from evo import EvoException
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit

# Score of a metric that cannot be computed (failed trial).
BAD_SCORE = 100.0


def is_bad(score):
    """True if score is the BAD_SCORE sentinel."""
    return bool(np.isclose(float(score), BAD_SCORE, rtol=0.0, atol=1e-9))


def _est_is_degenerate(traj_est):
    """True if Umeyama alignment is impossible: non-finite or collapsed positions."""
    pos = traj_est.positions_xyz
    return not np.isfinite(pos).all() or np.linalg.norm(np.std(pos, axis=0)) < 1e-8


def ate_rmse(traj_ref, traj_est):
    """Translation RMSE [m] after Sim(3) alignment, or BAD_SCORE."""
    associated = sync.associate_trajectories(traj_ref, traj_est)
    if _est_is_degenerate(associated[1]):
        return BAD_SCORE

    try:
        rmse = main_ape.ape(*associated, pose_relation=PoseRelation.translation_part,
                            align=True, correct_scale=True).stats["rmse"]
    except (np.linalg.LinAlgError, EvoException):
        return BAD_SCORE
    return float(rmse) if np.isfinite(rmse) else BAD_SCORE


def rpe(traj_ref, traj_est):
    """(RPE-t [m/frame], RPE-r [deg/frame]) RMSE over consecutive frames after Sim(3) alignment, or BAD_SCORE."""
    associated = sync.associate_trajectories(traj_ref, traj_est)
    if _est_is_degenerate(associated[1]):
        return BAD_SCORE, BAD_SCORE
    traj_ref, traj_est = associated

    try:
        traj_est.align(traj_ref, correct_scale=True)
        rpe_t, rpe_r = [float(main_rpe.rpe(traj_ref, traj_est, pose_relation=relation, delta=1,
                                           delta_unit=Unit.frames).stats["rmse"])
                        for relation in (PoseRelation.translation_part, PoseRelation.rotation_angle_deg)]
    except (np.linalg.LinAlgError, EvoException):
        return BAD_SCORE, BAD_SCORE
    if not (np.isfinite(rpe_t) and np.isfinite(rpe_r)):
        return BAD_SCORE, BAD_SCORE
    return rpe_t, rpe_r
