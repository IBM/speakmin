# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import matplotlib.pyplot as plt

# Original log-normal distribution parameters
# mu, sigma = 0.5, 0.5
mu, sigma = 0.5, 0.5
num_neu_res = 1000

# Generate initial log-normal samples
tau_samples = np.random.lognormal(mu, sigma, num_neu_res)

# Define the boundaries for resampling
lower_bound = 0.0
upper_bound = 3.0

# Separate samples into bins
inside_bounds = (tau_samples >= lower_bound) & (tau_samples <= upper_bound)
outside_bounds = ~inside_bounds

# Resample from the region inside the bounds to replace some of the tail samples
num_resample = np.sum(outside_bounds)
resampled_values = np.random.lognormal(mu, sigma, num_resample)

# Clip resampled values to ensure they fall within the desired range
resampled_values = np.clip(resampled_values, lower_bound, upper_bound)

# Combine the original inside-bound samples with the new resampled values
tau_samples_resampled = np.concatenate((tau_samples[inside_bounds], resampled_values))

# Ensure better representation of each tau value
tau_values = np.array([200, 400, 600, 800, 1000, 1200, 1400])
bins = np.linspace(0.0, 3.0, len(tau_values) + 1)
tau_bins = np.digitize(tau_samples_resampled, bins)
tau_bins = np.clip(tau_bins - 1, 0, len(tau_values) - 1)
tau_array_hetero = tau_values[tau_bins]

# Echo mean value
mean_actual = np.mean(tau_array_hetero)
print(f"Actual mean: {mean_actual}")

# Echo unique tau value and counts
unique, counts = np.unique(tau_array_hetero, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Value {value}: {count} samples")

# Plot histgram of the distribution
plt.hist(tau_array_hetero, bins=np.linspace(200, 1400, 8), edgecolor='black', density=True, alpha=0.6, color='b')
plt.title('Log-Normal Distribution with Adjusted Mean')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Plot mean value lines and text
plt.axvline(mean_actual, color='r', linestyle='dashed', linewidth=1)
plt.text(mean_actual, plt.ylim()[1]*0.9, f'Mean: {mean_actual:.2f}', color='r')

# Tight layout
plt.tight_layout()

# Save as PNG
plt.savefig('./log_normal_histogram.png')
plt.show()

print("Histogram saved to log_normal_histogram.png")
