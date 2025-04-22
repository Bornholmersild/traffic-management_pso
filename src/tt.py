import xml.etree.ElementTree as ET
import random 
import numpy as np
tree = ET.parse("sumo/random_routes.rou.xml")  # Load your .rou.xml file
root = tree.getroot()
import os

#os.system('python3 $SUMO_HOME/tools/randomTrips.py -n sumo/sumo_network.net.xml -r sumo/random_routes.rou.xml -b 0 -e 750 -p 1')

th = [[3, 5], [10, 11], [0, 1]]
th = np.array(th)  # Convert to a NumPy array

# Find the index of the row with the minimum value
min_row_index = np.argmin(np.min(th, axis=1))

# Extract the row with the minimum value
min_row = th[min_row_index]

print(min_row)  # Output: [0, 1]
'''
for vehicle in root.findall("vehicle"):
    veh_id = vehicle.get("id")
    depart_time = vehicle.get("depart")
    print(f"Vehicle {veh_id} departs at {depart_time}")


for vehicle in root.findall("vehicle"):
    veh_id = vehicle.get("id")
    route = vehicle.find("route").get("edges")
    print(f"Vehicle {veh_id} follows route: {route}")

vehicle_routes = {}

for vehicle in root.findall("vehicle"):
    veh_id = vehicle.get("id")
    route = vehicle.find("route").get("edges").split()
    vehicle_routes[veh_id] = route

print(vehicle_routes)

import random

for vehicle in root.findall("vehicle"):
    vehicle.set("depart", str(random.uniform(0, 50)))  # Random time 0-50s

tree.write("updated_routes.rou.xml")  # Save new file
'''