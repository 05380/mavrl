import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_trimmed_min_max(values):
    lower_bound = np.percentile(values, 5)
    upper_bound = np.percentile(values, 95)
    return lower_bound, upper_bound

def compute_min_max(values, remove_outliers):
    if remove_outliers:
        return compute_trimmed_min_max(values)
    else:
        return min(values), max(values)

def plot_data(file_path, column_index, remove_outliers=False):
    # Load the data
    data = pd.read_csv(file_path, delimiter=' ', header=None)
    
    # Split the data into 10 groups
    group_size = 50
    num_groups = 3
    groups = [data.iloc[i:i+group_size, :] for i in range(0, num_groups*group_size, group_size)]
    
    # Unique x values
    unique_x_values = sorted(data.iloc[:, 0].unique())
    
    # Compute the mean, min, and max y values for each unique x value across all groups
    mean_y_values = []
    min_y_values = []
    max_y_values = []
    
    for x in unique_x_values:
        y_values_at_x = [group[group.iloc[:, 0] == x].iloc[0, column_index] for group in groups if x in group.iloc[:, 0].values]
        mean_y_values.append(sum(y_values_at_x) / len(y_values_at_x))
        min_y, max_y = compute_min_max(y_values_at_x, remove_outliers)
        min_y_values.append(min_y)
        max_y_values.append(max_y)
    
    # Plotting
    line, = plt.plot(unique_x_values, mean_y_values)
    plt.fill_between(unique_x_values, min_y_values, max_y_values, alpha=0.2)
    plt.xlabel('Iterations')
    plt.ylabel('Success Rate')
    plt.grid(True)
    
    return line

# Usage example
plt.figure(figsize=(10, 6))
line1 = plot_data('./matlab/success_rate_101.csv', 1, remove_outliers=True)
line2 = plot_data('./matlab/success_rate_010.csv', 1, remove_outliers=True)
line3 = plot_data('./matlab/success_rate_110.csv', 1, remove_outliers=True)
plt.legend([line1, line2, line3], ['101', '010', '110'])
plt.title('Easy')

plt.figure(figsize=(10, 6))
line1 = plot_data('./matlab/success_rate_101.csv', 2, remove_outliers=True)
line2 = plot_data('./matlab/success_rate_010.csv', 2, remove_outliers=True)
line3 = plot_data('./matlab/success_rate_110.csv', 2, remove_outliers=True)
plt.legend([line1, line2, line3], ['101', '010', '110'])
plt.title('Hard')
plt.show()