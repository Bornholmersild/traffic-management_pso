import traci
import numpy as np
import os
import random
import csv
import time
# Load your .rou.xml file
import xml.etree.ElementTree as ET
tree = ET.parse("sumo/random_routes.rou.xml") 
root = tree.getroot()


class PSO_TrafficOptimizer:
    def __init__(self, file_network, sim_iterations, num_particles=50, iterations_max=50, w_max=0.5, w_min=0.1, 
                 c1=2.0, c2=2, phase_min=5, phase_max=50, lamda_factor=0.5, gui_on=False):
        ''' Parameters for simulation
        '''
        self.ST = sim_iterations                # Total simulation time
        self.TT = 0                             # Total trip time of all vehicles
        self.SW = 0                             # Sum of waiting time of all vehicles
        self.V = 0                              # Number of vehicles that reached their destination
        self.NV = 0                             # Number of vehicles which did not reach their destination  
        self.P = 0                              # proportion of colors in the phase duration
        self.existing_veh = set()
        self.depart_veh = {}
        self.waiting_veh_time = {}
        self.wait_sum = 0
        self.vehicle_routes = {}
        self.shuffled_veh_ids = []

        ''' Parameters for PSO algorithm
        param:
        num_particles: solution for the optimization problem
        num_iterations: number of iterations to run the optimization algorithm
        w_max: maximum inertia weight
        w_min: minimum inertia weight
            w: is the trade-off between exploration and exploitation
        c1: relative effect on the personal particle
        c2: relative effect on the global best particle
        phase_min: minimum phase duration
        phase_max: maximum phase duration
        '''
        self.num_particles = num_particles
        self.iterations_max = iterations_max
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.lambda_factor = lamda_factor

        if gui_on:
            #self.sumoCmd = ["sumo-gui", "--time-to-teleport", "-n", file_network[0]]
            self.sumoCmd = ["sumo-gui", "--time-to-teleport", "-1", "-n", file_network[0], "-r", file_network[1]]
        else:
            #self.sumoCmd = ["sumo", "--time-to-teleport", "-n", file_network[0]]
            self.sumoCmd = ["sumo", "--time-to-teleport", "-1", "-n", file_network[0], "-r", file_network[1]]

        self.extract_tls()
        self.particles, self.velocities = self.initialize_particles()
        self.personal_best = np.full((self.num_particles, self.num_lights * self.num_phases), np.inf)
        self.personal_best_fitness = np.full(self.num_particles, np.inf)
        self.global_best = np.full(self.num_particles, np.inf)
        self.global_best_fitness = np.inf
        #self.extract_vehicles(file_network[1])
        

    def run(self, independent_run):
            
        file_path = f'output/logging_run{independent_run+1}.csv'
        if os.path.exists(file_path):
            #raise FileExistsError(f"Error: The file {file_path} already exists. Execution stopped.")
            os.remove(file_path)
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')  # Use tab as separator for better spacing
            writer.writerow(["Iteration", "Global Fitness", "Arrived Vehicles", "Non Arrived Vehicles", "Trip Time", "Waiting Time"])

            for iteration in range(self.iterations_max):
                try:
                    traci.start(self.sumoCmd)
                except Exception as e:
                    print("Error starting SUMO:", e)
                    return 

                if iteration > 0:
                    self.apply_particle_to_sumo()
                else:
                    self.init_random_particle_to_sumo()
                
                #self.debug_traffic_light()

                for step in range(self.ST):
                    #self.add_random_vehicles(sim_step=step, max_vehicles=2, summon_step=2)
                    traci.simulationStep()
                    self.pso_param()
                    #time.sleep(0.005)
                
                print("Simulation ended")
                
                self.update_particles(iteration)
                
                
                print(f"---------------------Iteration: {iteration}---------------------")
                print(f"Run: {run+1}/{indenpendent_run}")
                print(f"Simulation time: {self.ST}")
                print(f"Vehicles that reached their destination: {np.sqrt(self.V)}")
                print(f"Total trip time of all vehicles: {self.TT}")
                print(f"Sum of waiting time of all vehicles: {self.SW}")
                print(f"Number of vehicles which did not reach their destination: {self.NV}")
                print(f"Proportion of colors in the phase duration: {self.P}")
                print(f"Fitness cost: {self.global_best_fitness}")
                print("Global best phases (reshaped):\n", self.global_best.reshape(self.num_lights, self.num_phases))
                
                traci.close()
                
            
                writer.writerow([f"{iteration:<10}", 
                    f"{self.global_best_fitness:<15f}", 
                    f"{np.sqrt(self.V):<19}", 
                    f"{self.NV:<23.1f}", 
                    f"{self.TT:<8.1f}", 
                    f"{self.SW:<6.1f}"])
                
                self.reset_param()
        
    def reset_param(self):
        self.TT = 0                             # Total trip time of all vehicles
        self.SW = 0                             # Sum of waiting time of all vehicles
        self.V = 0                              # Number of vehicles that reached their destination
        self.NV = 0                             # Number of vehicles which did not reach their destination  
        self.P = 0                              # proportion of colors in the phase duration
        self.existing_veh = set()
        self.depart_veh = {}
        self.waiting_veh_time = {}
        self.wait_sum = 0

    def pso_param(self):
        '''
        param:
        V:  vehicles to reach their destination
        TT: global trip time of all vehicles
        ST: simulation time
        SW: sum of waiting time of all vehicles
        V²: squared to weight the importance of this term
        NV: number of vehicles which did not reach their destination
        '''      
        arrived_number = traci.simulation.getArrivedNumber()        # Extract number of vehicles that arrived
        departed_number = traci.simulation.getDepartedNumber()      # Extract number of vehicles that departed
        current_time = traci.simulation.getTime()                   # Extract current simulation time

        if arrived_number > 0:
            self.V += arrived_number                                # Update total number of vehicles that reached their destination
            self.TT += self.track_arrived_vehicles(current_time)    # Update total trip time of all vehicles that arrived
            
        if departed_number > 0:
            self.track_departed_vehicles()                          # Update the vehicles that departed in a list with their departure time
            
        for veh_id in traci.vehicle.getIDList():                    
            wait_time = traci.vehicle.getWaitingTime(veh_id)        # Accumulate the waiting time of each vehicle for each simulation time step
            if wait_time > 0:
                self.SW += 1
            
        if current_time == self.ST:
            self.NV = len(self.existing_veh)                        # Extract the number of vehicles which did not reach their destination in the end of simulation
            self.V *= self.V                                        # Squared V to priority its importance
            
            #for id, val in self.depart_veh.items():                 # Sum all travel time for existing vehicles that have not left the map after simulation time
            #    self.TT += val
        
    def track_arrived_vehicles(self, current_time):
        '''
        param:
            arrived_number is given by : traci.simulation.getArrivedNumber()
            current_time is given by : traci.simulation.getTime()
        return:
            trip_time: total trip time of all vehicles that arrived
        '''
        trip_time = 0
        for veh_id in traci.simulation.getArrivedIDList():         # Iterate over arrived vehicles
            assert veh_id in self.existing_veh                     # Ensure it is in the existing vehicles
            trip_time += current_time - self.depart_veh[veh_id]    # Calculate trip time
            self.existing_veh.remove(veh_id)                       # Remove vehicle from existing vehicles
            del self.depart_veh[veh_id]
        
        return trip_time

    def track_departed_vehicles(self):
        '''
        param:
            arrived_number is given by : traci.simulation.getDepartedNumber()
        '''
        for veh_id in traci.simulation.getDepartedIDList():            # Iterate over departed vehicles
            assert veh_id not in self.existing_veh                     # Ensure vehicle is not already in the map
            self.existing_veh.add(veh_id)                              # Add vehicle to existing vehicles
            self.depart_veh[veh_id] = traci.vehicle.getDeparture(veh_id)
        
    def track_traffic_light_colors(self, particle):
        P = 0
        phase_idx = 0

        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)

            for phase in logic[0].phases:  # Get first logic (default program)
                if phase.state.count('y') > 0:
                    continue

                duration = particle[phase_idx]                 # Get duration of phase
                green_count = phase.state.count('G')      # Count green signals
                #green_count += phase.state.count('GGGgggrrr')
                red_count = phase.state.count('r')        # Count red signals
                if red_count > 0:
                    P += duration * (green_count / red_count)  # Avoid division by zero
                else:
                    P += duration * (green_count / 1) 
                    
                
                phase_idx += 1

        return P 
    
    def initialize_particles(self):
        """ Initialize swarm with random traffic light durations."""
        #particles = np.random.randint(self.phase_min, self.phase_max, size=(self.num_particles, self.num_lights * self.num_phases))
        particles = np.random.randint(self.phase_min, self.phase_max, size=(self.num_particles, self.num_lights * self.num_phases))
        velocities = np.zeros((self.num_particles, self.num_lights * self.num_phases))
        #print("Particle 0 (reshaped):\n", particles[0].reshape(self.num_lights, self.num_phases))
        #print("Particle 0 (reshaped):\n", velocities[0].reshape(self.num_lights, self.num_phases))

        return particles, velocities

    def fitness_function(self, particle):
        """ Run SUMO simulation and return fitness score."""
        self.P = self.track_traffic_light_colors(particle)

        if self.V == 0:
            self.V = 0.00001

        #return 1/self.V
        return (self.TT + self.SW + (self.NV * self.ST))/(self.V + self.P)
        #return self.SW / (self.V + self.P)

    def locate_best(self, idx, current_fitness):
        # Compare fitness scores and update personal_best if current is better
        if current_fitness < self.personal_best_fitness[idx]:  # Minimization problem                
            self.personal_best[idx] = self.particles[idx]
            self.personal_best_fitness[idx] = current_fitness
    
    def apply_particle_to_sumo(self):
        phase_index = 0
        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            clip_phases = []

            for phase in logic.phases:  # Get first logic (default program)
                if phase.state.count('y') > 0:
                    duration = 3

                else:
                    # Set new traffic light program
                    duration = self.global_best[phase_index]  # Get the duration from particle
                    phase_index += 1
                
                clip_phases.append(traci.trafficlight.Phase(duration, phase.state))
            
            traci.trafficlight.setProgramLogic(tls, traci.trafficlight.Logic(logic.programID, 0, logic.type, clip_phases))

    def init_random_particle_to_sumo(self):
        phase_index = 0

        min_index = np.argmin(np.sum(self.particles, axis=1))
        random_phase = self.particles[min_index].astype(int).tolist()
        
        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            clip_phases = []

            for phase in logic.phases:  # Get first logic (default program)
                if phase.state.count('y') > 0:
                    duration = 3
            
                else:
                    # Set new traffic light program
                    duration = random_phase[phase_index]  # Get the duration from particle
                    phase_index += 1
                
                clip_phases.append(traci.trafficlight.Phase(duration, phase.state))
                
            
            traci.trafficlight.setProgramLogic(tls, traci.trafficlight.Logic(logic.programID, 0, logic.type, clip_phases))

    def update_particles(self, iteration):  # iteration should be 1 in the beginning?
        """ Update particle positions and velocities."""
        w = self.w_max - (self.w_max - self.w_min) * (iteration / self.iterations_max)          
        
        for i in range(self.num_particles):                             # Iterate over all particles
            fitness = self.fitness_function(self.particles[i])    # Evaluate objective function
            self.locate_best(i, fitness)                            # Update personal best
            #print(f"Particle {i} - Fitness: {fitness}")


        personal_best_idx = np.argmin(self.personal_best_fitness)

        if self.personal_best_fitness[personal_best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[personal_best_idx]
            self.global_best_fitness = self.personal_best_fitness[personal_best_idx]

        for i in range(self.num_particles):
            r1, r2 = np.random.rand(), np.random.rand()                 # random variable from a uniform distribuion between [0, 1]
            #print(f"personal_best - particles: {self.personal_best[i] - self.particles[i]}")
            #print(f"global_best - particles: {self.global_best - self.particles[i]}")

            self.velocities[i] = (w * self.velocities[i] +              # Update velocity 
                                        self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                        self.c2 * r2 * (self.global_best - self.particles[i]))
            
            temp_particle = self.particles[i] + self.velocities[i]             # Temperal particle position before floor/ceil rounding

            round_prob = np.random.rand(self.num_lights * self.num_phases)   # One random probability per phase
            temp_particle = np.where(round_prob <= self.lambda_factor,
                                        np.floor(temp_particle),
                                        np.ceil(temp_particle))
            
            # print(f"Particle {i} - Old position: {self.particles[i]}")

            self.particles[i] = np.clip(temp_particle, self.phase_min, self.phase_max)  # Update position - limited between phase_min and phase_max
            
            # print(f"Particle {i} - New position: {self.particles[i]}")
            # print(f"Particle {i} - velocity: {self.velocities[i]}")
        
    def extract_tls(self):
        traci.start(self.sumoCmd)
        phase_temp = 0
        self.num_lights = traci.trafficlight.getIDCount()
        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)
            phase_temp += len(logic[0].phases)
        
        self.num_phases = phase_temp
        traci.close()

    def add_random_vehicles(self, sim_step, max_vehicles=1, summon_step=2):
        # Depart time witn delay
        # Random route
        # Number of vechicles
        if sim_step == 0:
            veh_ids = list(self.vehicle_routes.keys())
            random.shuffle(veh_ids)
            self.shuffled_veh_ids = veh_ids

        if np.mod(sim_step, summon_step) != 0:
            return
        
        for i in range(max_vehicles):
            spawn_time = sim_step + random.randint(0, 1)  # Random delay (0-5s)
            veh_id = self.shuffled_veh_ids[sim_step + i]
            route = self.vehicle_routes[veh_id]
            
            traci.vehicle.add(vehID=veh_id, 
                              routeID="",
                              depart=spawn_time)
            
            # Assign edges to the vehicle
            traci.vehicle.setRoute(veh_id, route)
            

    def extract_vehicles(self):
        for vehicle in root.findall("vehicle"):
            veh_id = vehicle.get("id")
            route = vehicle.find("route").get("edges").split()
            self.vehicle_routes[veh_id] = route
        

if __name__ == "__main__":
    #network = ["sumo/sumo_network.net.xml", "sumo/traffic_config.rou.xml"]
    #network = ["sumo/sumo_network.net.xml", "sumo/random_routes.rou.xml"]
    #network = ["sumo/sumo_complex.net.xml", "sumo/random_complex.rou.xml"]
    #network = ["sumo/sumo_network.net.xml"]
    network = ["sumo/sumo_network_spiral.net.xml", "sumo/random_routes_spiral.rou.xml"]
    
    indenpendent_run = 1
    for run in range(indenpendent_run):
        pso = PSO_TrafficOptimizer(
                                network,
                                sim_iterations=500,
                                num_particles=10,
                                iterations_max=10,
                                w_max=0.9,
                                w_min=0.4,
                                c1=2,
                                c2=2,
                                phase_min=5,
                                phase_max=50,
                                lamda_factor=0.5,
                                gui_on=False
                                )
        pso.run(run)
    
    '''
    sumoCmd = ["sumo-gui", "-n", network[0], "-r", network[1]]
    traci.start(sumoCmd)
    for i in range(500):
        traci.simulationStep()
        veh = traci.vehicle.getIDList()
        print("{veh} \n")
    
    traci.close()'''





