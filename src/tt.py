import numpy as np
import os
import csv
from sumo_simulation import TrafficSimulation

rng = np.random.default_rng(1)

round_prob = np.random.rand(self.num_phases)                # One random probability per phase
temp_particle = np.where(round_prob <= self.lamda_factor,
                            np.floor(temp_particle),
                            np.ceil(temp_particle))