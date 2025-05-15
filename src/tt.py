import numpy as np
import os
import csv
from sumo_simulation import TrafficSimulation

rng = np.random.default_rng(1)

list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])



for _ in range(5):
    rng.shuffle(list)
    print(list)

