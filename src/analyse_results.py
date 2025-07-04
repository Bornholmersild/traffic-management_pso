import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from sklearn.linear_model import LinearRegression
import seaborn as sns
import glob

# Configure Matplotlib to use LaTeX and set font size
# Set global font size
plt.rc('font', size=18)

x_step = 5  # step size for x-axis plot

def plot_metrics_from_csv(file_path, save_path):
    """
    Plots metrics from a single CSV file.

    Args:
        file_path (str): Path to the CSV file.
        save_path (str): Path to graph save
    """

    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    # Load the CSV file
    df = pd.read_csv(file_path, delimiter='\t')

    # Plot 1: Fitness
    fitness_data = df[df['Metric'] == 'fitness']
    plt.figure(figsize=(8, 6))  # 8, 6
    plt.plot(fitness_data['Iteration'], fitness_data['Value'], label='Fitness', color='blue')
  
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Iterations')
    plt.legend()
    plt.xlim(0, fitness_data['Iteration'].max())
    plt.gca().xaxis.set_major_locator(MultipleLocator(x_step))
    plt.tight_layout()
    fitness_save_path = os.path.join(save_path, "fitness_one_run.png")
    plt.savefig(fitness_save_path)
    plt.close()
    
    # Plot 2: Arrived/Non-Arrived Vehicles and Total Trip/Wait Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # First subplot: Arrived and Non-Arrived Vehicles
    arrived_data = df[df['Metric'] == 'arrived_vehicles']
    non_arrived_data = df[df['Metric'] == 'non_arrived_vehicles']
    ax1.plot(arrived_data['Iteration'], arrived_data['Value'], label='Arrived Vehicles', color='green')

    ax1.plot(non_arrived_data['Iteration'], non_arrived_data['Value'], label='Non-Arrived Vehicles', color='red')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Cars')
    ax1.set_title('Arrived and Non-Arrived Vehicles Over Iterations')
    ax1.set_xlim(0, arrived_data['Iteration'].max())
    ax1.xaxis.set_major_locator(MultipleLocator(x_step))
    ax1.legend()

    # Second subplot: Total Trip Time and Total Wait Time
    trip_time_data = df[df['Metric'] == 'total_trip_time']
    wait_time_data = df[df['Metric'] == 'total_wait_time']
    ax2.plot(trip_time_data['Iteration'], trip_time_data['Value'], label='Total Trip Time', color='purple')
  
    ax2.plot(wait_time_data['Iteration'], wait_time_data['Value'], label='Total Wait Time', color='orange')
 
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Seconds')
    ax2.set_title('Total Trip Time and Total Wait Time Over Iterations')
    ax2.set_xlim(0, trip_time_data['Iteration'].max())
    ax2.xaxis.set_major_locator(MultipleLocator(x_step))
    ax2.legend()

    # Adjust layout and show the second figure
    combined_save_path = os.path.join(save_path, "combined_metrics_one_run.png")
    plt.tight_layout()
    plt.savefig(combined_save_path)  # Save the combined plot
    plt.close()

def plot_metrics_from_multiple_csvs(base_path, save_path, num_files):
    """
    Iterates over multiple CSV files and plots metrics for all runs on the same graph.

    Args:
        base_path (str): Base path to the CSV files (e.g., 'output/logging_run').
        save_path (str): Path to save the combined graphs.
        num_files (int): Number of CSV files to iterate over.
    """
    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    # Plot 1: Fitness
    plt.figure(figsize=(10, 6))
    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        # Filter fitness data
        fitness_data = df[df['Metric'] == 'global_fitness']
        plt.plot(fitness_data['Iteration'], fitness_data['Value'], label=f'Run {i}')

    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Iterations (All Runs)')
    plt.xlim(0, 19)
    plt.gca().xaxis.set_major_locator(MultipleLocator(x_step))
    plt.legend()
    fitness_save_path = os.path.join(save_path, "fitness_all_runs.png")
    plt.savefig(fitness_save_path)
    plt.close()
    
    # Plot 2: Arrived and Non-Arrived Vehicles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))  #10, 10

    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        # First subplot: Arrived and Non-Arrived Vehicles
        arrived_data = df[df['Metric'] == 'arrived_vehicles']
        non_arrived_data = df[df['Metric'] == 'non_arrived_vehicles']
        
        ax1.plot(arrived_data['Iteration'], arrived_data['Value'], label=f'Arrived Vehicles (Run {i})')

        ax2.plot(non_arrived_data['Iteration'], non_arrived_data['Value'], label=f'Non-Arrived Vehicles (Run {i})')

    # Finalize first subplot
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Vehicles')
    ax1.set_title('Arrived Vehicles Over Iterations (All Runs)')
    ax1.set_xlim(0, arrived_data['Iteration'].max())
    ax1.xaxis.set_major_locator(MultipleLocator(x_step))
    ax1.legend()

    # Finalize second subplot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Vehicles')
    ax2.set_title('Non-Arrived Vehicles over Iterations (All Runs)')
    ax2.set_xlim(0, non_arrived_data['Iteration'].max())
    ax2.xaxis.set_major_locator(MultipleLocator(x_step))
    ax2.legend()

    # Save the combined plot
    combined_save_path = os.path.join(save_path, "Arrived_nonArrived_vehicles.png")
    plt.tight_layout()
    plt.savefig(combined_save_path)
    plt.close()

    # Plot 3: Total Trip and Total Wait Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))  # 10,10

    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        trip_time_data = df[df['Metric'] == 'total_trip_time']
        wait_time_data = df[df['Metric'] == 'total_wait_time']

        # First subplot: Total Trip Time
        ax1.plot(trip_time_data['Iteration'], trip_time_data['Value'], label=f'Total Trip Time (Run {i})')
        
        # Second subplot: Total Wait Time
        ax2.plot(wait_time_data['Iteration'], wait_time_data['Value'], label=f'Total Wait Time (Run {i})')

    # Finalize first subplot
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Seconds')
    ax1.set_title('Total Trip Time Over Iterations (All Runs)')
    ax1.set_xlim(0, trip_time_data['Iteration'].max())
    ax1.xaxis.set_major_locator(MultipleLocator(x_step))
    ax1.legend()

    # Finalize second subplot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Seconds')
    ax2.set_title('Total Wait Time Over Iterations (All Runs)')
    ax2.set_xlim(0, wait_time_data['Iteration'].max())
    ax2.xaxis.set_major_locator(MultipleLocator(x_step))
    ax2.legend()

    combined_save_path = os.path.join(save_path, "trip_wait_time.png")
    plt.tight_layout()
    plt.savefig(combined_save_path)
    plt.close()

def plot_mean_metrics_from_multiple_csvs(base_path, save_path, num_files):
    """
    Iterates over multiple CSV files and plots metrics for all runs on the same graph.

    Args:
        base_path (str): Base path to the CSV files (e.g., 'output/logging_run').
        save_path (str): Path to save the combined graphs.
        num_files (int): Number of CSV files to iterate over.
    """
    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    # Plot 1: Fitness
    plt.figure(figsize=(10, 6))
    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        # Filter fitness data
        fitness_data = df[df['Metric'] == 'global_fitness']
        fitness_data = fitness_data[0:40]
        plt.plot(fitness_data['Iteration'], fitness_data['Value'], label=f'Run {i}')

    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Iterations (All Runs)')
    plt.xlim(0, 39)
    plt.gca().xaxis.set_major_locator(MultipleLocator(x_step))
    plt.legend()
    fitness_save_path = os.path.join(save_path, "fitness_all_runs.png")
    plt.savefig(fitness_save_path)
    plt.close()
    
    # Plot 2: Arrived and Non-Arrived Vehicles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))  #10, 10

    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        # First subplot: Arrived and Non-Arrived Vehicles
        arrived_data = df[df['Metric'] == 'arrived_vehicles']
        non_arrived_data = df[df['Metric'] == 'non_arrived_vehicles']
        arrived_data = arrived_data[0:40]
        non_arrived_data = non_arrived_data[0:40]
        
        ax1.plot(arrived_data['Iteration'], arrived_data['Value'], label=f'Arrived Vehicles (Run {i})')

        ax2.plot(non_arrived_data['Iteration'], non_arrived_data['Value'], label=f'Non-Arrived Vehicles (Run {i})')

    # Finalize first subplot
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Vehicles')
    ax1.set_title('Arrived Vehicles Over Iterations (All Runs)')
    ax1.set_xlim(0, 39)
    ax1.xaxis.set_major_locator(MultipleLocator(x_step))
    ax1.legend()

    # Finalize second subplot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Vehicles')
    ax2.set_title('Non-Arrived Vehicles over Iterations (All Runs)')
    ax2.set_xlim(0, 39)
    ax2.xaxis.set_major_locator(MultipleLocator(x_step))
    ax2.legend()

    # Save the combined plot
    combined_save_path = os.path.join(save_path, "Arrived_nonArrived_vehicles.png")
    plt.tight_layout()
    plt.savefig(combined_save_path)
    plt.close()

    # Plot 3: Total Trip and Total Wait Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))  # 10,10

    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        trip_time_data = df[df['Metric'] == 'total_trip_time']
        wait_time_data = df[df['Metric'] == 'total_wait_time']
        trip_time_data = trip_time_data[0:40]
        wait_time_data = wait_time_data[0:40]

        # First subplot: Total Trip Time
        ax1.plot(trip_time_data['Iteration'], trip_time_data['Value'], label=f'Total Trip Time (Run {i})')
        
        # Second subplot: Total Wait Time
        ax2.plot(wait_time_data['Iteration'], wait_time_data['Value'], label=f'Total Wait Time (Run {i})')

    # Finalize first subplot
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Seconds')
    ax1.set_title('Total Trip Time Over Iterations (All Runs)')
    ax1.set_xlim(0, 39)
    ax1.xaxis.set_major_locator(MultipleLocator(x_step))
    ax1.legend()

    # Finalize second subplot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Seconds')
    ax2.set_title('Total Wait Time Over Iterations (All Runs)')
    ax2.set_xlim(0, 39)
    ax2.xaxis.set_major_locator(MultipleLocator(x_step))
    ax2.legend()

    combined_save_path = os.path.join(save_path, "trip_wait_time.png")
    plt.tight_layout()
    plt.savefig(combined_save_path)
    plt.close()

def plot_metrics_from_multiple_csvs_combined(base_path, save_path, num_files):
    """
    Iterates over multiple CSV files and plots metrics for all runs on the same graph.

    Args:
        base_path (str): Base path to the CSV files (e.g., 'output/logging_run').
        save_path (str): Path to save the combined graphs.
        num_files (int): Number of CSV files to iterate over.
    """
    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    # Plot 1: Fitness
    plt.figure(figsize=(8, 6))
    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        # Filter fitness data
        fitness_data = df[df['Metric'] == 'global_fitness']
        plt.plot(fitness_data['Iteration'], fitness_data['Mean'], label=f'Run {i}')

    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Iterations (All Runs)')
    plt.legend()
    fitness_save_path = os.path.join(save_path, "fitness_all_runs.svg")
    plt.savefig(fitness_save_path)
    plt.close()

    # Plot 2: Arrived/Non-Arrived Vehicles and Total Trip/Wait Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        # First subplot: Arrived and Non-Arrived Vehicles
        arrived_data = df[df['Metric'] == 'arrived_vehicles']
        non_arrived_data = df[df['Metric'] == 'non_arrived_vehicles']
        ax1.plot(arrived_data['Iteration'], arrived_data['Mean'], label=f'Arrived Vehicles (Run {i})')

        ax1.plot(non_arrived_data['Iteration'], non_arrived_data['Mean'], label=f'Non-Arrived Vehicles (Run {i})')

        # Second subplot: Total Trip Time and Total Wait Time
        trip_time_data = df[df['Metric'] == 'total_trip_time']
        wait_time_data = df[df['Metric'] == 'total_wait_time']
        ax2.plot(trip_time_data['Iteration'], trip_time_data['Mean'], label=f'Total Trip Time (Run {i})')

        ax2.plot(wait_time_data['Iteration'], wait_time_data['Mean'], label=f'Total Wait Time (Run {i})')

    # Finalize first subplot
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Cars')
    ax1.set_title('Arrived and Non-Arrived Vehicles Over Iterations (All Runs)')
    step_size = 2  # Example step size
    ax1.xaxis.set_major_locator(MultipleLocator(step_size))

    ax1.legend()

    # Finalize second subplot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Seconds')
    ax2.set_title('Total Trip Time and Total Wait Time Over Iterations (All Runs)')
    ax2.legend()

    # Save the combined plot
    combined_save_path = os.path.join(save_path, "combined_metrics_all_runs.svg")
    plt.tight_layout()
    plt.savefig(combined_save_path)
    plt.close()



def plot_metrics_from_multiple_csvs_combined_custom(base_path, save_path, num_files):
    """
    Iterates over multiple CSV files and plots metrics for all runs on the same graph.

    Args:
        base_path (str): Base path to the CSV files (e.g., 'output/logging_run').
        save_path (str): Path to save the combined graphs.
        num_files (int): Number of CSV files to iterate over.
    """
    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    # Plot 1: Arrived vehicles
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 18))

    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"
        print(f"Processing: {file_path}")

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        if i == 2:
            fitness_data = df[df['Metric'] == 'fitness']
            ax1.plot(fitness_data['Iteration'], fitness_data['Mean'], label=f'Fitness (Run {i})')
            ax1.fill_between(fitness_data['Iteration'], 
                         fitness_data['Mean'] - fitness_data['Std'], 
                         fitness_data['Mean'] + fitness_data['Std'], 
                         alpha=0.1, label='Std Dev')

        # Plot 1: Fitness
        arrived_data = df[df['Metric'] == 'arrived_vehicles']
        wait_data = df[df['Metric'] == 'total_wait_time']

        ax2.plot(arrived_data['Iteration'], arrived_data['Mean'], label=f'Arrived Vehicles (Run {i})')
        ax3.plot(wait_data['Iteration'], wait_data['Mean'], label=f'Wait Time (Run {i})')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fitness')  # Adjust the value of labelpad as needed
    ax1.set_title('Fitness Over Iterations (Single Run)')
    ax1.legend()
    ax1.set_xlim(0, arrived_data['Iteration'].max())
    ax1.xaxis.set_major_locator(MultipleLocator(x_step))

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Cars')  # Adjust the value of labelpad as needed
    ax2.set_title('Arrived Vehicles Over Iterations (All Runs)')
    ax2.set_xlim(0, arrived_data['Iteration'].max())
    ax2.legend()
    ax2.xaxis.set_major_locator(MultipleLocator(x_step))
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Seconds')
    ax3.set_title('Total Wait Time Over Iterations (All Runs)')
    ax3.set_xlim(0, wait_data['Iteration'].max())
    ax3.legend()
    ax3.xaxis.set_major_locator(MultipleLocator(x_step))
    # Apply scientific notation to the y-axis
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

    plt.tight_layout()

    
    save_path = os.path.join(save_path, "wait_p_trip.svg")
    plt.savefig(save_path)
    plt.close()


def calculate_mean_std(base_path, num_files=5):
    """
    Iterates over multiple CSV files and calculates mean and std for the last iteration of each metric.

    Args:
        base_path (str): Base path to the CSV files (e.g., 'output/logging_run').
        num_files (int): Number of CSV files to iterate over - Default 5
    """

    print(f"---------{base_path}---------")
    metrics = ['fitness', 'arrived_vehicles', 'non_arrived_vehicles', 'total_trip_time', 'total_wait_time']
    results = {metric: np.zeros(num_files) for metric in metrics}

    for i in range(1, num_files + 1):
        file_path = f"{base_path}{i}.csv"

        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        for metric in metrics:
            metric_data = df[df['Metric'] == metric]
            
            # Extract last iteration data - Validation data
            last_iteration_data = metric_data[metric_data['Iteration'] == metric_data['Iteration'].max()]
            results[metric][i - 1] = last_iteration_data['Value'].tolist()[0]
            
            # Extract secound last - Last training data
            #second_last_iteration = metric_data['Iteration'].iloc[-1]  # Get the second last iteration
            #second_last_iteration_data = metric_data[metric_data['Iteration'] == second_last_iteration]
            # Extract mean value
            #results[metric][i - 1] = second_last_iteration_data['Value'].tolist()[0]

    # Calculate and print mean and std for each metric
    for metric in metrics:
        print(f"{metric} - Mean: {np.mean(results[metric]):}, Std: {np.std(results[metric]):}")

def fit_exponential(X, Y):
    """
    Fits an exponential model Y = a * exp(b * X) using linear regression on log-transformed Y.

    Args:
        X (array-like): Independent variable.
        Y (array-like): Dependent variable.

    Returns:
        tuple: (a, b) where Y ≈ a * exp(b * X).
    """
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    # Avoid log(0) by filtering or adding epsilon
    Y = np.where(Y <= 0, 1e-10, Y)
    log_Y = np.log(Y)

    model = LinearRegression()
    model.fit(X, log_Y)

    b = model.coef_[0]
    a = np.exp(model.intercept_)

    return a, b    

def plot_boxplots_for_metrics(file_path):

    files = glob.glob(file_path + "*.csv")
    df = pd.concat([pd.read_csv(file, delimiter='\t') for file in files], ignore_index=True)
    # Filter only 'fitness' metric
    fitness_data = df[df['Metric'] == 'fitness']

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='Iteration',
        y='Mean',
        data=fitness_data,
        showfliers=True,  # Show outliers
        palette='coolwarm'
    )

    plt.title('Box Plot of Fitness Across Iterations', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Fitness Mean Value', fontsize=14)
    plt.xticks(rotation=90)  # Rotate if needed
    plt.tight_layout()
    plt.show()

def plot_hist_for_metrics_across_runs(base_paths, num_files, save_path="output"):
    metrics = ['arrived_vehicles', 'non_arrived_vehicles', 'total_trip_time', 'total_wait_time']
    xlabel_txt = ['V', 'Simpel', 'Complex', 'Complex_with_P']
    xlabel_txt = ['1/V²', 'SW/V²', 'Complex']
    overall_mean = {metric: [] for metric in metrics}
    overall_std = {metric: [] for metric in metrics}

    for base_path in base_paths:
        print(f"---------{base_path}---------")
        results = {metric: np.zeros(num_files) for metric in metrics}

        for i in range(1, num_files + 1):
            file_path = f"{base_path}{i}.csv"
            # Load the CSV file
            df = pd.read_csv(file_path, delimiter='\t')

            lowest_fitness_iteration = df[df['Metric'] == 'fitness']['Iteration'][df[df['Metric'] == 'fitness']['Value'].idxmin()]

            for metric in metrics:
                metric_data = df[df['Metric'] == metric]
                # Extract last 5 iteration data
                #last_iteration_data = metric_data[metric_data['Iteration'] == metric_data['Iteration'].max()]
                #results[metric][i - 1] = last_iteration_data['Value'].tolist()[0]
                
                iteration_data = metric_data[metric_data['Iteration'] == lowest_fitness_iteration]
                results[metric][i - 1] = iteration_data['Value'].tolist()[0]
                

        # Calculate overall mean and std for the last 5 values across all files for this base_path
        for metric in metrics:
            overall_mean[metric].append(np.mean(results[metric]))
            overall_std[metric].append(np.std(results[metric]))

    first_group_metrics = metrics[:2]  # First two metrics
    second_group_metrics = metrics[2:]  # Last two metrics

    first_group_min = min([min(overall_mean[metric]) for metric in first_group_metrics])
    first_group_max = max([max(overall_mean[metric]) for metric in first_group_metrics])

    second_group_min = min([min(overall_mean[metric]) for metric in second_group_metrics])
    second_group_max = max([max(overall_mean[metric]) for metric in second_group_metrics])

    # Create subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(40, 25))
    x_positions = np.arange(len(base_paths))  # Positions for the bars

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x_positions, overall_mean[metric], yerr=overall_std[metric], color=['skyblue', 'orange', 'green', 'red'], capsize=5, alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{xlabel_txt[i]}' for i in range(len(base_paths))], fontsize=20)
        ax.set_title(f'{metric.capitalize()}', fontsize=24)
        ax.set_ylabel('Mean Value', fontsize=20)

        #         # Set y-axis limits based on the group
        if metric in first_group_metrics:
            ax.set_ylim(0, first_group_max+200)
        elif metric in second_group_metrics:
            ax.set_ylim(0, second_group_max+20000)

    plt.tight_layout()
    save_path = os.path.join(save_path, "globalbest_metric_across_cost.png")
    plt.savefig(save_path)
    plt.close()
    plt.show()

def plot_hist_for_metrics_across_particles(base_paths, num_files, save_path="output"):
    metrics = ['fitness', 'arrived_vehicles', 'non_arrived_vehicles', 'total_trip_time', 'total_wait_time']
    xlabel_txt = "Run"
    global_results = {metric: np.zeros(num_files) for metric in metrics}
    global_fitness_lines = np.zeros(num_files)

    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    for i in range(1, num_files + 1):
        file_path = f"{base_paths}{i}.csv"
        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        # Find the iteration where 'fitness' has the lowest value
        lowest_fitness_iteration = df[df['Metric'] == 'fitness']['Iteration'][df[df['Metric'] == 'fitness']['Value'].idxmin()]
        
        for metric in metrics:
            metric_data = df[df['Metric'] == metric]
            # Extract last 5 iteration data
            
            #last_iteration_data = metric_data[metric_data['Iteration'] == df['Iteration'].max()]
            #global_results[metric][i - 1] = last_iteration_data['Value'].tolist()[0]
            # Find the iteration where 'fitness' has the lowest value
            iteration_data = metric_data[metric_data['Iteration'] == lowest_fitness_iteration]
            global_results[metric][i - 1] = iteration_data['Value'].tolist()[0]
    
        # Extract global_fitness for horizontal lines
        global_fitness_data = df[df['Metric'] == 'global_fitness']
        
        global_fitness_lines[i - 1] = global_fitness_data['Value'].iloc[-1]
        print(global_fitness_lines[i-1])

    first_group_metrics = metrics[1:3]  # First two metrics
    second_group_metrics = metrics[3:5]  # Last two metrics

    first_group_min = min([min(global_results[metric]) for metric in first_group_metrics])
    first_group_max = max([max(global_results[metric]) for metric in first_group_metrics])

    second_group_min = min([min(global_results[metric]) for metric in second_group_metrics])
    second_group_max = max([max(global_results[metric]) for metric in second_group_metrics])

    # Create subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(40, 25))
    x_positions = np.arange(num_files)  # Positions for the bars

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x_positions, global_results[metric], capsize=5, color=['skyblue', 'orange', 'green', 'red', 'purple'])
        # Add a horizontal line at a fixed value (e.g., 50) for a specific bar (e.g., the second bar)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'Run {i}' for i in range(1, num_files + 1)], fontsize=20)
        ax.set_title(f'{metric.capitalize()} Across Runs', fontsize=24)
        ax.set_ylabel('Mean Value', fontsize=20)

                        # Set y-axis limits based on the group
        if metric in first_group_metrics:
            ax.set_ylim(0, first_group_max) # first_group_min
        elif metric in second_group_metrics:
            ax.set_ylim(0, second_group_max) # second_group_min

        if idx == 0:
            for j in range(num_files):
                fixed_value = global_fitness_lines[j]  # The y-value for the horizontal line
                specific_bar_index = j  # Index of the bar (0-based)
                ax.hlines(y=fixed_value, xmin=specific_bar_index - 0.4, xmax=specific_bar_index + 0.4, colors='black', linestyles='dashed', label='Threshold')


    # Add a common x-label for all subplots
    fig.text(0.5, 0.04, xlabel_txt, ha='center', fontsize=14)
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(save_path, "metrics_bar_plots.png")
    plt.savefig(save_path)
    plt.close()

def plot_hist_for_identical_run(base_paths, select_run, save_path="output"):
    metrics = ['arrived_vehicles', 'non_arrived_vehicles', 'total_trip_time', 'total_wait_time']
    xlabel_txt = ['1/V²', 'SW/V²', 'Complex']
    results = {metric: [] for metric in metrics}
    

    for base_path in base_paths:
        print(f"---------{base_path}---------")

        file_path = f"{base_path}{select_run}.csv"
        
        # Load the CSV file
        df = pd.read_csv(file_path, delimiter='\t')

        lowest_fitness_iteration = df[df['Metric'] == 'fitness']['Iteration'][df[df['Metric'] == 'fitness']['Value'].idxmin()]

        for metric in metrics:
            metric_data = df[df['Metric'] == metric]
            iteration_data = metric_data[metric_data['Iteration'] == lowest_fitness_iteration]
            results[metric].append(iteration_data['Value'].tolist()[0])

    first_group_metrics = metrics[:2]  # First two metrics
    second_group_metrics = metrics[2:]  # Last two metrics

    first_group_max = max([max(results[metric]) for metric in first_group_metrics])
    second_group_max = max([max(results[metric]) for metric in second_group_metrics])

    # Create subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(40, 25))
    x_positions = np.arange(len(base_paths))  # Positions for the bars

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x_positions, results[metric], color=['skyblue', 'orange', 'green'], capsize=5, alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{xlabel_txt[i]}' for i in range(len(base_paths))], fontsize=20)
        ax.set_title(f'{metric.capitalize()}', fontsize=24)
        ax.set_ylabel('Mean Value', fontsize=20)

        #         # Set y-axis limits based on the group
        if metric in first_group_metrics:
            ax.set_ylim(0, first_group_max+200)
        elif metric in second_group_metrics:
            ax.set_ylim(0, second_group_max+20000)

    fig.suptitle(f"Metrics Across indenpendent run {select_run}", fontsize=30, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_path, f"Metrics_for_indenpendent_run{select_run}_across_cost_.png")
    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()
    plt.close()
    

def extract_validation_metrics(base_paths, save_path, select_run):
    '''
    Extract validation metrics from 5 indenpendent runs each with 5 validation runs for the same cost function
    '''
    metrics = ['fitness', 'arrived_vehicles', 'non_arrived_vehicles', 'total_trip_time', 'total_wait_time']
    global_results = {metric: [] for metric in metrics}
    global_fitness_lines = []

    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    for i in range(select_run, select_run + 1):  
        file_path = f"{base_paths}{i}.csv"
        df = pd.read_csv(file_path, delimiter='\t')
        
        for metric in metrics:
            metric_data = df[df['Metric'] == metric]
            values = metric_data['Value'].values

            global_results[metric].append(values)
    
        # Extract global_fitness for horizontal lines
        global_fitness_lines_df = (df[df['Metric'] == 'global_fitness'])
        global_fitness_lines.append(global_fitness_lines_df['Value'].iloc[-1])
    
    global_mean = {metric: np.mean(np.concatenate(global_results[metric])) for metric in metrics}
    global_std = {metric: np.std(np.concatenate(global_results[metric])) for metric in metrics}
    global_fitness_mean = np.mean(global_fitness_lines)
        
    return global_mean, global_std, global_fitness_mean

def plot_hist_for_validation(base_paths, select_run, save_path="output", num_cost_function=4):
    metrics = ['arrived_vehicles', 'non_arrived_vehicles', 'total_trip_time', 'total_wait_time']
    xlabel_txt = ['V', 'Simpel', 'Complex', 'Complex_with_P']
    xlabel_txt = ['1/V²', 'SW/V²', 'Complex']

    # Ensure the save folder exists
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, len(metrics), figsize=(40, 25))
    x_positions = np.arange(num_cost_function)  # Positions for the bars

    for i in range(num_cost_function):
        mean, std, global_fitness_value = extract_validation_metrics(base_paths[i], save_path, select_run)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            # Plot only at the correct x-position for this cost function
            ax.bar([i], [mean[metric]], yerr=[std[metric]], 
                   color=['skyblue', 'orange', 'green', 'red'][i % 4], capsize=5, alpha=0.7)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(xlabel_txt[:num_cost_function], fontsize=20)
            ax.set_title(f'{metric.capitalize()}', fontsize=24)
            ax.set_ylabel('Mean Value', fontsize=20)

    plt.tight_layout()
    #plt.savefig(os.path.join(save_path, "validation_across_cost.png"))
    plt.show()
    plt.close()


grid_V = "output/odense_VFitness/logging_run"
grid_simple = "output/odense_simpleFitness/logging_run"
grid_complex = "output/odense_complexFitness/logging_run"
grid_full = "output/odense_fullFitness/logging_run"
#plot_metrics_from_multiple_csvs(grid_V, "output/grid_5_tls_2_lanes_400_veh_VFitness", 5)
#plot_metrics_from_multiple_csvs(grid_simple, "output/grid_5_tls_2_lanes_400_veh_simpleFitness", 5)
#plot_metrics_from_multiple_csvs(grid_complex, "output/grid_5_tls_2_lanes_400_veh_complexFitness", 5)
#plot_metrics_from_multiple_csvs(grid_full, "output/grid_5_tls_2_lanes_400_veh_fullFitness", 5)
#plot_metrics_from_multiple_csvs(grid_simple, "output/odense_simpleFitness", 5)
#plot_metrics_from_csv("output/odense_simpleFitness/logging_run1.csv", "output/odense_simpleFitness")
#plot_metrics_from_multiple_csvs("output/grid_5_tls_2_lanes_400_veh_complexFitness/logging_run" , "output/grid_5_tls_2_lanes_400_veh_complexFitness", 1)

#calculate_mean_std(grid_V, 1)
#calculate_mean_std(grid_simple, 1)
#calculate_mean_std(grid_full, 1)

#print("----------------------Validation statistics----------------------")
#calculate_mean_std("output/grid_40_iterations/grid_5_tls_2_lanes_400_veh_complexFitness/logging_run", 5)
#calculate_mean_std("output/grid_40_iterations/grid_5_tls_2_lanes_400_veh_fullFitness/logging_run", 5)
#print("\n\n")
#calculate_mean_std("output/grid_5_tls_2_lanes_400_veh_simpleFitness/logging_run", 5)

# hist_base_path = ["output/grid_5_tls_2_lanes_400_veh_VFitness/logging_run", 
#                   "output/grid_5_tls_2_lanes_400_veh_simpleFitness/logging_run",
#                   "output/grid_5_tls_2_lanes_400_veh_complexFitness/logging_run", 
#                   "output/grid_5_tls_2_lanes_400_veh_fullFitness/logging_run"]


hist_base_path = [grid_V,
                  grid_simple,
                  grid_complex,
                  grid_full]

hist_base_path = [grid_V,
                  grid_simple,
                  grid_complex]

hist_base_path_validation = ["output/odense_VFitness_validation/validation_run",
                             "output/odense_simpleFitness_validation/validation_run"]

#plot_hist_for_metrics_across_particles(grid_V, 5, "output")
plot_hist_for_metrics_across_runs(hist_base_path, 5, "output")
#plot_hist_for_validation(hist_base_path, select_run=1, save_path="output", num_cost_function=3)
# for i in range(1, 6):
#     plot_hist_for_identical_run(hist_base_path, select_run=i, save_path="output")