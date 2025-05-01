import numpy as np
import os
import csv
from sumo_simulation import TrafficSimulation


class PSO_TrafficOptimizer:
    """Optimizes SUMO traffic light timings using Particle Swarm Optimization (PSO)."""

    def __init__(self, file_network, random_seed, sim_iterations, num_particles=50, iterations_max=50, 
                 w_max=0.5, w_min=0.1, c1=2.0, c2=2.0, phase_min=5, phase_max=50, 
                 lamda_factor=0.5, gui_on=False):
        
        self.simulation = TrafficSimulation(file_network, sim_iterations, random_seed, gui_on)
        
        # PSO Parameters
        self.num_particles = num_particles
        self.iterations_max = iterations_max
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.lamda_factor = lamda_factor
        self.seed = random_seed

        self.num_lights, self.num_phases = self.simulation.extract_tls()
        self.particles, self.velocities = self.initialize_particles()
        self.personal_best = np.full((self.num_particles, self.num_phases), np.inf)
        self.personal_best_fitness = np.full(self.num_particles, np.inf)
        min_particle_idx = np.argmin(np.min(self.particles, axis=1))
        self.global_best = self.particles[min_particle_idx]
        self.global_best_fitness = np.inf

    def initialize_particles(self):
        """Initializes swarm with random traffic light durations."""
        rng = np.random.default_rng(self.seed)
        particles = rng.integers(low=self.phase_min, high=self.phase_max, size=(self.num_particles, self.num_phases))
        velocities = np.zeros((self.num_particles, self.num_phases))
        return particles, velocities

    def run(self, independent_run, base_path_to_save):
        """
        Runs the PSO optimization process.
        
        Args:
            independent_run (int): Number of complete different runs
            base_path (str): Base path to save the CSV files (e.g., 'output/<folder>/logging_run').
        """
        os.makedirs(base_path_to_save, exist_ok=True)                   # Ensure the save folder exists
        log_file = f"{base_path_to_save}/logging_run{independent_run+1}.csv"

        if os.path.exists(log_file):
            os.remove(log_file)

        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Iteration', 'Metric', 'Value'])    # Before , std was added        #writer.writerow(["Iteration", "Global Fitness", "Arrived Vehicles", "Non Arrived Vehicles", "Trip Time", "Waiting Time"])

            for iteration in range(self.iterations_max):
                iteration_metrics = {'fitness': [], 'global_fitness': [], 'arrived_vehicles': [], 'non_arrived_vehicles': [], 'total_trip_time': [], 'total_wait_time': []}
                self.log_save_particle_phases(iteration, self.iterations_max, independent_run, f"{base_path_to_save}/particle_log.csv")  # Save the particle phases at key iterations

                for particle_idx in range(self.num_particles):     
                    self.simulation.start_sumo()
                    self.simulation.apply_particle_to_sumo(self.particles[particle_idx])
                    self.simulation.run_simulation()
                    fitness = self.evaluate(self.particles[particle_idx]) 
                    self.simulation.close_sumo()                    
                    self.update_best_particle(fitness, particle_idx)                          
                    self.update_particles(particle_idx, iteration)
                    self.log_to_terminal(independent_run, iteration, particle_idx, fitness)
                    self.collect_matrics(iteration_metrics, fitness)
                    self.reset_parameters()
                
                self.update_global_particle()
                self.log_to_csv(iteration, iteration_metrics, writer)
                if iteration == self.iterations_max - 1:        # Save the final global best particle - Which is not trained on
                    self.log_save_particle_phases(iteration + 1, self.iterations_max, independent_run, f"{base_path_to_save}/particle_log.csv")  # Save the particle phases at key iterations

            self.validation(self.global_best, self.iterations_max, writer)  # Validation of the best particle
    def evaluate(self, particle):
        """Computes the fitness function based on traffic metrics."""
        P = self.simulation.phase_proportion(particle)
        V = max(self.simulation.arrived_vehicles, 0.00001)  # Avoid division by zero

        #return 1/(V*V)
        #return (self.simulation.total_wait_time) / (V*V)
        #return (self.simulation.total_trip_time + self.simulation.total_wait_time + (self.simulation.non_arrived_vehicles * self.simulation.sim_iterations)) / (V*V)
        return (self.simulation.total_trip_time + self.simulation.total_wait_time + (self.simulation.non_arrived_vehicles * self.simulation.sim_iterations)) / (V*V + P)
    
    def update_best_particle(self, fitness, particle_idx):
        """Update current particle if fitness cost is lower"""
        if fitness < self.personal_best_fitness[particle_idx]:
            self.personal_best[particle_idx] = self.particles[particle_idx]
            self.personal_best_fitness[particle_idx] = fitness

    def update_particles(self, particle_idx, iteration):  # iteration should be 1 in the beginning?
        """ Update particle positions and velocities."""
        i = particle_idx
        w = self.w_max - (self.w_max - self.w_min) * (iteration / self.iterations_max)          
        r1, r2 = np.random.rand(), np.random.rand()                 # random variable from a uniform distribuion between [0, 1]            

        self.velocities[i] = (w * self.velocities[i] +              # Update velocity 
                                    self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                    self.c2 * r2 * (self.global_best - self.particles[i]))

        temp_particle = self.particles[i] + self.velocities[i]      # Temperal particle position before floor/ceil rounding

        round_prob = np.random.rand(self.num_phases)                # One random probability per phase
        temp_particle = np.where(round_prob <= self.lamda_factor,
                                    np.floor(temp_particle),
                                    np.ceil(temp_particle))

        self.particles[i] = np.clip(temp_particle, self.phase_min, self.phase_max)  # Update position - limited between phase_min and phase_max
    
    def update_global_particle(self):
        min_fitness_idx = np.argmin(self.personal_best_fitness)
        min_fitness_val = self.personal_best_fitness[min_fitness_idx]

        if min_fitness_val < self.global_best_fitness:
            self.global_best = self.particles[min_fitness_idx]
            self.global_best_fitness = min_fitness_val
    
    def collect_matrics(self, iteration_metrics, fitness):
        # Append the metrics to the iteration list
        iteration_metrics['fitness'].append(fitness)
        iteration_metrics['arrived_vehicles'].append(self.simulation.arrived_vehicles)
        iteration_metrics['non_arrived_vehicles'].append(self.simulation.non_arrived_vehicles)
        iteration_metrics['total_trip_time'].append(self.simulation.total_trip_time)
        iteration_metrics['total_wait_time'].append(self.simulation.total_wait_time)

    def log_to_csv(self, iteration, iteration_metrics, writer):
        """
        Logs the mean and standard deviation of metrics to a CSV file.

        Args:
            iteration (int): The current iteration number.
            iteration_metrics (dict): A dictionary containing lists of metrics for the iteration.
            writer (csv.writer): The CSV writer object to write the data.
        """
        '''Old Fasion Way - Consider mean and std for all particles
        for metric, values in iteration_metrics.items():
            if metric == 'global_fitness':
                writer.writerow([iteration, metric, self.global_best_fitness, 'N/A'])
            else:
                mean_val = np.mean(values)
                std_val = np.std(values)
                writer.writerow([iteration, metric, mean_val, std_val])
        '''

        global_best_idx = np.argmin(iteration_metrics['fitness'])

        for metric, value in iteration_metrics.items():
            if metric == "global_fitness":
                writer.writerow([iteration, metric, self.global_best_fitness])    
            else:
                val = value[global_best_idx]
                writer.writerow([iteration, metric, val])  # , std_val]) "Removed"


    def log_to_terminal(self, indenpendent_run, iteration, particle_idx, fitness):
        print(f"---------------------Independent run: {indenpendent_run+1}---------------------")
        print(f"Iteration: {iteration+1}")
        print(f"Particle: {particle_idx+1}")
        print(f"Vehicles Arrived: {self.simulation.arrived_vehicles}")
        print(f"Vehicles not Arrived: {self.simulation.non_arrived_vehicles}")
        print(f"Vehicles Total Trip Time: {self.simulation.total_trip_time}")
        print(f"Vehicles Total Wait Time: {self.simulation.total_wait_time}")
        print(f"Particle Current Fitness Cost: {fitness}")
        print(f"Particle Best Fitness Cost: {self.personal_best_fitness[particle_idx]}")
        #print(f"Particle current phases (reshaped): {self.particles[particle_idx].reshape(self.num_lights, self.num_phases)}")
        #print(f"Particle best phases (reshaped): {self.personal_best[particle_idx].reshape(self.num_lights, self.num_phases)}")

    def reset_parameters(self):
        """Resets tracking variables for a new iteration."""
        self.simulation.total_trip_time = 0
        self.simulation.total_wait_time = 0
        self.simulation.arrived_vehicles = 0
        self.simulation.non_arrived_vehicles = 0
        self.simulation.veh_stats = {}

    def log_save_particle_phases(self, iteration, max_iterations, independent_run, filename):
        # Define the key iteration milestones
        milestones = [
            0,  # First iteration
            max_iterations // 4,
            max_iterations // 2,
            3 * max_iterations // 4,
            max_iterations # Last iteration
        ]

            # Delete the file at the beginning (only once)
        if iteration == 0 and independent_run == 0 and os.path.exists(filename):
            os.remove(filename)

        # Check if the current iteration is a milestone
        if iteration in milestones:
            with open(filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                
                if iteration == 0:
                    # Write header only for the first iteration
                    writer.writerow([f"-------------------------Independent Run {independent_run+1}-------------------------"])

                writer.writerow([f"Iteration {iteration}"])

                init_phase = 0
                for phase in self.simulation.indices:
                    # Write the header for each traffic light
                    reshaped = self.global_best[init_phase:phase]
                    init_phase = phase
                    writer.writerow(["      "] + reshaped.tolist())

                writer.writerow([])  # Blank line between particles

    
    def validation(self, global_particle, max_iteration, writer):
        iteration_metrics = {'fitness': [], 'global_fitness': [], 'arrived_vehicles': [], 'non_arrived_vehicles': [], 'total_trip_time': [], 'total_wait_time': []}
        self.simulation.start_sumo()
        self.simulation.apply_particle_to_sumo(global_particle)
        self.simulation.set_seed(seed=10)
        self.simulation.run_simulation()
        fitness = self.evaluate(global_particle)
        self.simulation.close_sumo()
        self.collect_matrics(iteration_metrics, fitness)
        self.log_to_csv(max_iteration, iteration_metrics, writer)


   