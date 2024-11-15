import pickle
import random
from typing import Literal

import config as cfg
import numpy as np


class Path:
    def __init__(self):
        pass

    @staticmethod
    def get_or_create_paths(source: Literal["file", "generate"] = "file"):
        """Random path generator. Returns a list of random points within the 3D
        projection space.
        """
        if source == "file":
            with open("paths.pkl", "rb") as file:
                paths = pickle.load(file)
        elif source == "generate":
            paths = []
            for _ in range(cfg.NUMBER_OF_PATHS):
                path = []
                for j in range(cfg.SELECTION_THRESHOLD_DIVISOR):
                    x = random.uniform(-1, 1)
                    y = random.uniform(-1, 1)
                    z = random.uniform(-1, 1)
                    path.append([x, y, z])
                paths.append(path)
            paths = np.array(paths)
            with open("paths.pkl", "wb") as file:
                pickle.dump(paths, file)
        else:
            raise ValueError("Invalid source argument.")

        return paths
