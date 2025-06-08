import numpy as np

# we scale the depth value so that it can fit in np.uint16 and use image compression algorithms
DEPTH_IMG_SCALE = 16384

# simulation initialization settings
TABLE_HEIGHT = 0.74
OBJ_INIT_TRANS = np.array([0.51, 0.365, 0.82])
OBJ_RAND_RANGE = 1.3
OBJ_RAND_SCALE = 0.05

# clip the point cloud to a box
PC_MIN = np.array(
    [
        OBJ_INIT_TRANS[0] - OBJ_RAND_RANGE / 2,
        OBJ_INIT_TRANS[1] - OBJ_RAND_RANGE / 2,
        0.745,
    ]
)
PC_MAX = np.array(
    [
        OBJ_INIT_TRANS[0] + OBJ_RAND_RANGE / 2,
        OBJ_INIT_TRANS[1] + OBJ_RAND_RANGE / 2,
        0.895,
    ]
)
