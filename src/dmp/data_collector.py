import numpy as np
from typing import Union
from .dmp_proc import DMPProcessor

class DMPDataCollector:

    def __init__(self):
        self.t = []
        self.p = []

    def log(self, t: float, p: np.ndarray):
        self.t.append(t)
        self.p.append(p.tolist())

    def get(self):
        t = np.array(self.t)
        pos_traj = np.array(self.p).T
        return t, pos_traj
