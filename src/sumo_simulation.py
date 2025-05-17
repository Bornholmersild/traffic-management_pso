import traci
import numpy as np
import logging
import xml.etree.ElementTree as ET
import random
from sumolib.net import readNet
import traci.constants as tc

class TrafficSimulation:
    """Handles SUMO traffic simulation and collects relevant data."""
    
    def __init__(self, file_network, sim_iterations, random_seed, gui_on=False):
        self.sim_iterations = sim_iterations
        self.total_trip_time = 0
        self.total_wait_time = 0
        self.arrived_vehicles = 0
        self.non_arrived_vehicles = 0
        self.veh_stats = {} 
        self.vehicle_routes = {}
        self.shuffled_veh_ids = []
        self.file_network = file_network
        self.tree = ET.parse(file_network[1])       # Load the .rou.xml file 
        self.root = self.tree.getroot()
        self.seed = random_seed
        self.rng_veh = np.random.default_rng(self.seed)
        self.sumo_cmd = self._configure_sumo_cmd(file_network, gui_on)
        #self.last_egde_id = ['E0', '-24951719', '499165883#0', '495966519', '-9531177#2', '1042929708#1', '571750901#0', '-8032507#1', '-8028350#4', '-23594040#1', '1262417177', '-1287928531', '62074686', '-23242127']
        self.last_egde_id = ['E12', 'E17', '-E22', 'E21', '-E19', 'E18', '-E13', '-E11']
        self.depart_custom = 0
        self.veh_departed = 0

    def _configure_sumo_cmd(self, file_network, gui_on):
        """Configures SUMO command based on GUI preference."""
        mode = "sumo-gui" if gui_on else "sumo"
        #return [mode, "--time-to-teleport", "-1", "-n", file_network[0], "-r", file_network[1]]        
        #return [mode, "--time-to-teleport", "-1", "-c", file_network[0]]
        return [mode,
        "--time-to-teleport", "-1",
        "--waiting-time-memory", str(self.sim_iterations),
        "--no-step-log",
        "-n", file_network[0]
        ]
        
    def run_simulation(self):
        """Runs SUMO simulation and tracks vehicle data."""
        self.vehicle_routes, self.shuffled_veh_ids = self.extract_vehicles()               # Extract vehicles from the .rou.xml file
        max_veh = 5                                                                        # Number of vehicles to be spawned each simulation time
        for step in range(self.sim_iterations):
            if len(self.vehicle_routes)  > self.veh_departed:                                         # Pass if total amount of vehicles is exceeded
                if len(self.vehicle_routes) < self.veh_departed + max_veh:
                    max_veh = len(self.vehicle_routes) - self.veh_departed - 1
                self.add_random_vehicles(self.vehicle_routes, self.shuffled_veh_ids, max_veh=max_veh)
            traci.simulationStep()
            self.update_traffic_data(step)

    def start_sumo(self):
        try:
            traci.start(self.sumo_cmd)
        except Exception as e:
            logging.info("Simulation already running: %s", e)

    def close_sumo(self):
        """End simulation"""
        traci.close()

    def reset_sumo(self):
        #traci.load(["-n", self.file_network[0], "--time-to-teleport", "-1", "--waiting-time-memory", "-1", "--start"])
        traci.load(["-n", self.file_network[0], "--time-to-teleport", "-1", "--waiting-time-memory", str(self.sim_iterations), "--no-step-log", "--start"])


    def update_traffic_data(self, step):
        """Updates traffic statistics per step."""
        arrived_vehicles = traci.simulation.getArrivedNumber()
        departed_vehicles = traci.simulation.getDepartedNumber()
        
        current_time = step + 1

        if arrived_vehicles > 0:
            self.arrived_vehicles += arrived_vehicles
            self.track_arrived_vehicles(current_time)
            
        if departed_vehicles > 0:
            self.track_departed_vehicles()
        
        for egdes in self.last_egde_id:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(egdes)
            if vehicles_on_edge:
                for veh_id in vehicles_on_edge:
                    if self.veh_stats[veh_id]['arrive'] == False:
                        self.veh_stats[veh_id]['wait_time'] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                        self.veh_stats[veh_id]['arrive'] = True
        
        if current_time == self.sim_iterations:
            for veh_id in traci.vehicle.getIDList():
                self.non_arrived_vehicles += 1

    def track_arrived_vehicles(self, current_time):
        """Calculates total trip time for arrived vehicles."""
        trip_time = 0
        wait_time = 0

        for veh_id in traci.simulation.getArrivedIDList():
            trip_time += current_time - self.veh_stats[veh_id]["departure_time"]
            wait_time += self.veh_stats[veh_id]["wait_time"]
            self.veh_stats.pop(veh_id, None)

        self.total_trip_time += trip_time
        self.total_wait_time += wait_time

    def track_departed_vehicles(self):
        """Records vehicle departure times."""
        for veh_id in traci.simulation.getDepartedIDList():
            self.veh_stats[veh_id] = {
                "departure_time": traci.vehicle.getDeparture(veh_id),
                "wait_time": 0.0,
                "arrive" : False
            }


    def extract_tls(self):
        """Extract total number of traffic lights and phases"""
        traci.start(self.sumo_cmd)
        num_phase_temp = 0
        num_tls_temp = 0
        num_light_temp = traci.trafficlight.getIDCount()
        self.indices = np.full(num_light_temp, 0)

        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)
            
            for phase in logic[0].phases:
                if phase.state.count('y') == 0:
                    num_phase_temp += 1
            
            self.indices[num_tls_temp] = num_phase_temp
            num_tls_temp += 1
            
        return num_light_temp, num_phase_temp

    def phase_proportion(self, particle):
        """Calculate phase proportion considering green and red phase"""
        P = 0
        phase_idx = 0

        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)

            for phase in logic[0].phases:  # Get first logic (default program)
                if phase.state.count('y') > 0:
                    continue

                duration = particle[phase_idx]                 # Get duration of phase
                green_count = phase.state.count('G')      # Count green signals
                red_count = phase.state.count('r')        # Count red signals

                if red_count > 0:
                    P += duration * (green_count / red_count)  # Avoid division by zero
                else:
                    P += duration * (green_count / 1) 
                    
                phase_idx += 1

        return P 
    
    def apply_particle_to_sumo(self, particle):
        """Uses global best from previous results to set traffic light"""
        phase_index = 0
        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            clip_phases = []

            for phase in logic.phases:  # Get first logic (default program)
                if phase.state.count('y') > 0:
                    duration = 3

                else:
                    duration = particle[phase_index]  # Get the duration from particle
                    phase_index += 1
                
                clip_phases.append(traci.trafficlight.Phase(duration, phase.state))
            
            traci.trafficlight.setProgramLogic(tls, traci.trafficlight.Logic(logic.programID, 0, logic.type, clip_phases))

            # Set initial phase duration **immediately**
            current_phase = traci.trafficlight.getPhase(tls)  # Get the current phase index
            traci.trafficlight.setPhaseDuration(tls, clip_phases[current_phase].duration)   

    def add_random_vehicles(self, vehicle_routes, shuffled_veh_ids, max_veh):
        '''
        Add vehicles to the simulation at a specific simulation step.

        Param:
            vehicle_routes   : Dictionary with vehicle IDs as keys and their routes as values - Extracted using : extract_vehicles()
            shuffled_veh_ids : List of shuffled vehicle IDs - Extracted using : extract_vehicles()
            sim_step         : Simulation step to add vehicles
            shuffle          : Boolean to shuffle vehicle IDs
        '''    
        
        for _ in range(max_veh):
            
            veh_id = shuffled_veh_ids[self.veh_departed]
            route = vehicle_routes[veh_id]
            
            traci.vehicle.add(vehID=veh_id, 
                                routeID="",
                                depart=self.depart_custom)
            self.veh_departed += 1
            # Assign edges to the vehicle
            traci.vehicle.setRoute(veh_id, route)
        
        self.depart_custom += 1
            
    def extract_vehicles(self):
        '''
        return:
            vehicle_routes   : Extract vehicles from the .rou.xml file
            shuffled_veh_ids : Create a dictionary with vehicle IDs and their routes and shuffle them
        '''
        # Extract vehicles
        vehicle_routes = {}
        for vehicle in self.root.findall("vehicle"):
            veh_id = vehicle.get("id")
            route = vehicle.find("route").get("edges").split()
            vehicle_routes[veh_id] = route
        
        # Shuffle vehicle IDs
        shuffled_veh_ids = list(vehicle_routes.keys())
        self.rng_veh.shuffle(shuffled_veh_ids)

        #print(f"Shuffled vehicle IDs: \n {shuffled_veh_ids}")
        
        return vehicle_routes, shuffled_veh_ids
    
    def set_seed(self, seed):
        """
        Set the new seed for the simulation.

        Param:
            seed : seed value
        """
        self.seed = seed

    def generate_random_routes(self, net_file, rou_file, num_vehicles=100):
        net = readNet(net_file)
        
        # Grid world
        depart = ['-E12', '-E17', 'E22', '-E21', 'E19', '-E18', 'E13', 'E11']
        #arrive = ['E12', 'E17', '-E22', 'E21', '-E19', 'E18', '-E13', '-E11']
        
        # Odense World
        #depart = ['851348185#0', '24951719', '-499165883#1', '-495966519', '9531177#0', '-1042929708#2', '-571750901#2', '8032507#0', '8028350#3', '23594040#0', '1262417180', '451958405', '-658173492', '23242127']
        arrive = self.last_egde_id

        with open(rou_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n\n')
            f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
            f.write('    <vType id="car" accel="2.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="50"/>\n\n')

            for i in range(num_vehicles):
                start_edge = random.choice(depart)
                end_edge = random.choice(arrive)

                while start_edge == end_edge:
                    end_edge = random.choice(arrive)

                f.write(f'    <vehicle id="veh{i}" type="car" depart="{i}">\n')
                f.write(f'        <route edges="{start_edge} ')
                
                # Now compute a valid path using net.findShortestPath (Dijkstra)
                connect_route, _ = net.getShortestPath(net.getEdge(start_edge), net.getEdge(end_edge))
                existing_ids = {start_edge, end_edge}

                for edge in connect_route:
                    edge_id = edge.getID()
                    if edge_id not in existing_ids:
                        f.write(f'{edge_id} ')
                
                f.write(f'{end_edge}"/>\n')
                f.write(f'    </vehicle>\n')

            f.write('</routes>')

