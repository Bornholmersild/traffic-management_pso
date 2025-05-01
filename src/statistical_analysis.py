import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro             # Statistical test for normality
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp

def plot_histogram(data, title='Histogram', xlabel='Value', ylabel='Frequency'):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def check_normality(data):
    #data = np.array([your_values_here])

    # Shapiro-Wilk Test
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    if p > 0.05:
        print('Probably Normal')
    else:
        print('Probably Not Normal')


def kruskal_test(metric, grouped_data):
    # Perform Kruskal-Wallis test
    stat, p = kruskal(*grouped_data)  # Replace with your actual groups
    print(f'\nMetric: {metric}')
    print('Kruskal-Wallis test statistic=%.5f, p=%.5f' % (stat, p))

    if p > 0.05:
        print('Probably no significant difference between runs')
    else:
        print('Probably a significant difference between runs')

        # Post-hoc Dunn's test
        flat_data = np.concatenate(grouped_data)
        group_labels = np.concatenate([[i]*len(group) for i, group in enumerate(grouped_data)])
        df = pd.DataFrame({'Performance': flat_data, 'Group': group_labels})

        dunn_results = sp.posthoc_dunn(df, val_col='Performance', group_col='Group', p_adjust='bonferroni')
        print("\nPost-hoc Dunn's test results (p-values):")
        print(dunn_results)

cost_paths = ["output/grid_5_tls_2_lanes_400_veh_VFitness/logging_run", 
                  "output/grid_5_tls_2_lanes_400_veh_simpleFitness/logging_run",
                  "output/grid_5_tls_2_lanes_400_veh_complexFitness/logging_run", 
                  "output/grid_5_tls_2_lanes_400_veh_fullFitness/logging_run"
]
num_files = 5
metrics = ['arrived_vehicles', 'non_arrived_vehicles', 'total_trip_time', 'total_wait_time']

cost_results = {metric: [] for metric in metrics}

for base_path in cost_paths:                            # Iterate over cost functions
        print(f"---------{base_path}---------")
        results = {metric: np.zeros(num_files) for metric in metrics}

        for i in range(1, num_files + 1):               # Iterate over indenpendent runs
            file_path = f"{base_path}{i}.csv"
            # Load the CSV file
            df = pd.read_csv(file_path, delimiter='\t')

            for metric in metrics:                      # Iterate over performance metrices
                metric_data = df[df['Metric'] == metric]
                # Extract last 5 iteration data
                last_iteration_data = metric_data[metric_data['Iteration'] == metric_data['Iteration'].max()]
                # Calculate mean and std for the last 5 values
                results[metric][i - 1] = last_iteration_data['Mean'].tolist()[0]
        
        for metric in metrics:
            cost_results[metric].append(results[metric])  # Append this cost function's data


for metric in metrics:
    grouped_data = cost_results[metric]
    kruskal_test(metric, grouped_data)