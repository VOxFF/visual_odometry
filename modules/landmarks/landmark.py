from dataclasses import dataclass, field
import numpy as np


@dataclass
class Landmark:
    id: int
    xyz_world: np.ndarray       # (3,) position in world frame
    covariance: np.ndarray      # (3,3) uncertainty ellipsoid in world frame
    descriptors: list = field(default_factory=list)   # one np.ndarray per observation
    observations: int = 1
    last_seen_frame: int = 0
