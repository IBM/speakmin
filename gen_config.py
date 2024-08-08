import numpy as np
import json
import os

# Define the dimensions for the weights and parameters
num_neu_in = 16
# num_neu_res = 472
num_neu_res = 496
num_neu_out = 10
# num_neu_res = 412
# num_neu_out = 100
# num_neu_res = 462
# num_neu_out = 50
# num_neu_res = 492
# num_neu_out = 20
num_neu_bias = 10

# Function to quantize weights to 8-bit based on custom range
def quantize_weights(weights, min_val, max_val):
    # Scale weights to [0, 255] range based on min and max values
    quantized = np.round((weights - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    # Scale back to original range
    dequantized = quantized / 255 * (max_val - min_val) + min_val
    return dequantized

# Generate random weights for W_res
W_res = np.random.uniform(-1, 1, (num_neu_res, num_neu_res))
sparsity = 0.3
mask = np.random.rand(num_neu_res, num_neu_res) > sparsity
W_res[mask] = 0
np.fill_diagonal(W_res, 0)
rho_W_res = max(abs(np.linalg.eigvals(W_res)))
# W_res = W_res / rho_W_res * 0.95
W_res = W_res / rho_W_res * 1.25
# spectral_radius = 1.25

# Calculate in_scale for W_in
# in_scale = 1 / rho_W_res * 0.95
in_scale = 1 / rho_W_res * 1.25
W_in = np.random.uniform(-1, 1, (num_neu_in, num_neu_res)) * in_scale
sparsity = 0.3
mask = np.random.rand(num_neu_in, num_neu_res) > sparsity
W_in[mask] = 0
W_out = np.random.uniform(-1, 1, (num_neu_res, num_neu_out)) * in_scale
W_bias = np.random.uniform(-1, 1, (num_neu_bias, num_neu_res + num_neu_out)) * in_scale * 0
print("scaling factor is ", in_scale)
W_fb = np.random.uniform(-1, 1, (num_neu_res, num_neu_out))>0


W_in_quantized = W_in.tolist()
W_res_quantized = W_res.tolist()
W_out_quantized = W_out.tolist()
W_bias_quantized = W_bias.tolist()
W_fb = W_fb.astype(bool).tolist()

# 로그-노멀 분포의 파라미터
mu, sigma = 0, 0.5
# mu, sigma = 0.5, 0.5
# mu, sigma = 0.5, 0.5
# mu, sigma = 0, 0.75
# mu, sigma = 0.5, 0.5
num_samples = num_neu_res

# Generate initial log-normal samples
tau_samples = np.random.lognormal(mu, sigma, num_samples)

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
tau_values = np.array([100, 200, 300, 400, 500, 600, 700]) * 5
# tau_values = np.array([500, 1000, 2500, 5000, 7500, 10000, 20000]) * 2
# tau_values = np.array([500, 1000, 2500, 5000, 7500, 10000, 20000])
# tau_values = np.array([5000, 7500, 10000, 12500, 15000, 17500, 20000])/10
# tau_values = np.array([200, 400, 600, 800, 1000, 1200, 1400])
bins = np.linspace(0.0, 3.0, len(tau_values) + 1)
tau_bins = np.digitize(tau_samples_resampled, bins)
tau_bins = np.clip(tau_bins - 1, 0, len(tau_values) - 1)
tau_array_hetero = tau_values[tau_bins]

# 결과의 평균을 확인
mean_actual = np.mean(tau_array_hetero)
print(f"Actual mean: {mean_actual}")

# 각 구간별 개수 프린트
unique, counts = np.unique(tau_array_hetero, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Value {value}: {count} samples")

# Set homogeneous tau value (e.g., tau = 400)
# tau_homo_value = 50000.0
tau_homo_value = 100000.0
tau_array_homo = np.full(num_neu_res, tau_homo_value)
print(tau_homo_value)

# Choose between homogeneous and heterogeneous tau values
tau_type = "hetero"  # Change to "homo" for homogeneous tau values

if tau_type == "hetero":
    tau_array = tau_array_hetero
else:
    tau_array = tau_array_homo

# Define the weights dictionary with quantized weights
weights = {
    "W_in": W_in_quantized,
    "W_res": W_res_quantized,
    "W_out": W_out_quantized,
    "W_bias": W_bias_quantized,
    "W_fb": W_fb,
    "tau": tau_array.tolist()
}

# Define the simulation parameters dictionary
parameters = {
    "T_sim": 1000000000,
    "t_delay": 3,
    "V_init": 0.0,
    "tau": 500.0,
    "V_th": 1.0, 
    "V_reset": 0.0,
    "t_ref": 4,
    "N_in": num_neu_in,
    "N_res": num_neu_res,
    "N_out": num_neu_out,
    "N_bias": num_neu_bias,
    "SG_window": 0.5,
}

# Paths to the JSON files in the src directory
weights_path = os.path.join('src', 'init_weights.json')
parameters_path = os.path.join('src', 'init_parameter.json')
tau_path = os.path.join('src', 'init_taus.json')

# Save the weights to the weights.json file
with open(weights_path, 'w') as f:
    json.dump(weights, f, indent=4)
print(f"Weights file saved to {weights_path}")

# Save the parameters to the config.json file
with open(parameters_path, 'w') as f:
    json.dump(parameters, f, indent=4)
print(f"Parameters file saved to {parameters_path}")

# Save the tau values to the init_taus.json file
tau_dict = {"tau": tau_array.tolist()}
with open(tau_path, 'w') as f:
    json.dump(tau_dict, f, indent=4)
print(f"Tau file saved to {tau_path}")

# Check the distribution of tau values
tau_distribution = {tau: list(tau_array).count(tau) for tau in tau_values}
print(f"Tau distribution: {tau_distribution}")

print(f"{parameters}")
# print(f"{sparsity}")
