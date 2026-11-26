from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 4096

# VO config
_C.PATCHES_PER_FRAME = 192
_C.REMOVAL_WINDOW = 22
_C.OPTIMIZATION_WINDOW = 10
_C.PATCH_LIFETIME = 13

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 15.0

# camera motion model
_C.MOTION_DAMPING = 0.5

cfg = _C
