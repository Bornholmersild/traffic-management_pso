import xml.etree.ElementTree as ET
import random 
import numpy as np
import os
from sumo_simulation import TrafficSimulation
from pso_traffic_optimizer import PSO_TrafficOptimizer
import pandas as pd



file_network = ["sumo/world_odense.net.xml", "sumo/vehicles_odense.rou.xml"]
rou_path = "sumo/vehicles_odense.rou.xml"

TS = TrafficSimulation( file_network,
                        sim_iterations=500,
                        gui_on=False,
                        random_seed=2,
                        )
#"/home/nicklas/Sumo/2025-05-09-14-21-09/osm.net.xml.gz"
TS.generate_random_routes("sumo/world_odense.net.xml", rou_path, 6000)
