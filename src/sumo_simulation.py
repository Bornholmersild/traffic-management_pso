import traci
import numpy as np
import os
import csv
import logging

import traci.traciToHex

class TrafficSimulation:
    """Handles SUMO traffic simulation and collects relevant data."""
    
    def __init__(self, file_network, sim_iterations, gui_on=False):
        self.sim_iterations = sim_iterations
        self.total_trip_time = 0
        self.total_wait_time = 0
        self.arrived_vehicles = 0
        self.non_arrived_vehicles = 0
        self.veh_stats = {} 

        self.sumo_cmd = self._configure_sumo_cmd(file_network, gui_on)

    def _configure_sumo_cmd(self, file_network, gui_on):
        """Configures SUMO command based on GUI preference."""
        mode = "sumo-gui" if gui_on else "sumo"
        return [mode, "--time-to-teleport", "-1", "-n", file_network[0], "-r", file_network[1]]        

    def run_simulation(self):
        """Runs SUMO simulation and tracks vehicle data."""
        for step in range(self.sim_iterations):
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

        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getWaitingTime(veh_id) > 0:
                self.veh_stats[veh_id]['wait_time'] += 1 
        
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
                "wait_time": 0.0
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
