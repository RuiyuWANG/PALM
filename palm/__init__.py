import os

import numpy as np

from palm.utils.config_utils import load_config
from palm.utils.transform_utils import check_SE3

palm_config = load_config("meta/palm_dataset_meta")


RGB_OPS = palm_config["RGB_OPS"]
LOW_DIM_OPS = palm_config["LOW_DIM_OPS"]
LABEL_OPS = palm_config["LABEL_OPS"]
HOME_X = np.array(palm_config["HOME_X"])
WORKSPACE_X = np.array(palm_config["WORKSPACE_X"])

check_SE3(HOME_X)

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROTVEC_GAIN = 0.05
TRANSLATION_GAIN = 0.05

ROBOT_INIT_OFFSETS = {"Panda": None, "UR5": [0.1997, 0.0, 0.9995], "UR5_TipOnly": [0.1997, 0.0, 0.9995]}
OBJECT_WAYPOINTS = {
                    "lift_spam": [1], 
                    "insert_onto_single_square_peg_simple": [1], 
                    "rearrange_veges": [1, 8],
                    "take_lid_off_saucepan": [1],
                    "take_lid": [1],
                    "lift_lid": [1],
                    "pick_place_pepper": [1],
                    "pick_place_apple": [1],
                    "open_box": [1],
                    "close_box": [1],
                    }