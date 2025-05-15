import xml.etree.ElementTree as ET
import random 
import numpy as np
import os
from sumo_simulation import TrafficSimulation
from pso_traffic_optimizer import PSO_TrafficOptimizer
import pandas as pd

def extract_global_particle(filename):
    '''
    Extract global particle for particle_log file for five indenpendent runs

    '''
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    results = []
    current_run = None
    collecting = False
    values = []

    for i, line in enumerate(lines):
        # Detect new run
        if 'Independent Run' in line:
            current_run = line.strip()
            collecting = False
        # Detect iteration 200
        elif 'Iteration 100' in line:
            collecting = True
            values = []
        # Collect 5 lines of values after Iteration 40
        elif collecting:
            if line.strip() == "":
                continue
            if line.strip().startswith(','):
                # Extract numbers
                nums = [int(n) for n in line.strip().split(',') if n.isdigit()]
                values.extend(nums)
            if len(values) >= 35:  # Stop after 5 lines (approx 20 values)
                results.append((current_run, values.copy()))
                collecting = False

    return results

def extract_global_fitness(filename, num_files=5):
    metric = 'global_fitness'
    global_fitness = np.zeros(num_files)

    # Ensure the save folder exists
    os.makedirs(filename, exist_ok=True)

    for i in range(1, num_files + 1):
        file_path = f"{filename}{i}.csv"
        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')
        
        metric_data = df[df['Metric'] == metric]
        # Extract last 5 iteration data
        
        last_iteration_data = metric_data[metric_data['Iteration'] == df['Iteration'].max()]
        global_fitness[i - 1] = last_iteration_data['Value'].tolist()[0]
        
    return global_fitness




file_network = ["sumo/world_odense.net.xml", "sumo/vehicles_odense.rou.xml"]


pso = PSO_TrafficOptimizer(
                            file_network,
                            random_seed=1,
                            sim_iterations=1500,     # 500
                            num_particles=10,        # 10
                            iterations_max=200,       # 40
                            w_max=0.5,
                            w_min=0.1,
                            c1=2,
                            c2=2,
                            phase_min=10,
                            phase_max=40,
                            lamda_factor=0.5,
                            gui_on=False
                            )


# Validation for mutiple runs
# Example usage
file_path_to_particle_log = "output/odense_simpleFitness/particle_log.csv"
file_path_to_logging_run = "output/odense_simpleFitness/logging_run"

# Extract phase for global particle at 200 iteration. Stored in a array (1, 5)
# Extract global fitness for 5 independent runs. Stored in a array (1, 5)
data = extract_global_particle(file_path_to_particle_log)
global_fitness = extract_global_fitness(file_path_to_logging_run)

# Execute 5 validation runs for each independent run. Total of 25 validation runs
for indenpendent_run in range(5):
    # Extract global particle for each independent run
    global_particle = data[indenpendent_run][1]

    # Extract global fitness for each independent run
    global_cost = global_fitness[indenpendent_run]
    print(f"Indenpendent Run {indenpendent_run+1} \n Global Particle: {global_particle} \n Global Fitness: {global_cost}")

    # Execute validation run
    pso.validation_after_testing(global_particle=global_particle,
                                     iteration=indenpendent_run,
                                     base_path_to_save="output",
                                     global_fitness=global_cost
                                     )
                                     
