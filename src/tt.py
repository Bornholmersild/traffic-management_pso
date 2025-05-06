import xml.etree.ElementTree as ET
import random 
import numpy as np
import os
from sumo_simulation import TrafficSimulation




#os.system('python3 $SUMO_HOME/tools/randomTrips.py -n sumo/sumo_network.net.xml -r sumo/random_routes.rou.xml -b 0 -e 750 -p 1')
network = ["sumo/world_grid_5_tls_2_lanes.net.xml", "sumo/vehicles_grid_5_tls_2_lanesll.rou.xml"]
simulation = TrafficSimulation(network, 500, 1, True)

netfile = "sumo/world_grid_5_tls_2_lanes.net.xml"
roufile = "sumo/vehicles_grid_5_tls_2_lanesll.rou.xml"

simulation.generate_random_routes(netfile, roufile, num_vehicles=400)