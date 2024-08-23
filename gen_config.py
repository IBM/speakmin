import numpy as np
import json
import os

# Set the random seed for reproducibility
np.random.seed(0)

# Define the dimensions for the weights and parameters
num_neu_in = 16
num_neu_res = 496
num_class = 10
num_out_times = 1 
# 2, 4, 10
num_neu_out = num_class * num_out_times # 10
# 20, 40, 100

num_neu_bias = 10

# Function to quantize weights to 8-bit based on custom range
def quantize_weights(weights, min_val, max_val):
    quantized = np.round((weights - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    dequantized = quantized / 255 * (max_val - min_val) + min_val
    return dequantized

# Generate random weights for W_res
W_res = np.random.uniform(-1, 1, (num_neu_res, num_neu_res)) * 0.1
# sparsity = 0.3
# mask = np.random.rand(num_neu_res, num_neu_res) > sparsity
# W_res = W_res[mask]
# np.fill_diagonal(W_res, 0)
rho_W_res = max(abs(np.linalg.eigvals(W_res)))

# Calculate in_scale for W_in
in_scale = 1 / rho_W_res * 1.25
W_in = np.random.uniform(-1, 1, (num_neu_in, num_neu_res)) * 0.1

def xavier_init_uniform(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, size=(input_size, output_size))

def xavier_init_normal(input_size, output_size):
    stddev = np.sqrt(2 / (input_size + output_size))
    return np.random.normal(0, stddev, size=(input_size, output_size))

W_out = np.random.uniform(-1, 1, (num_neu_res, num_neu_out)) * 0.1
W_bias = np.random.uniform(-1, 1, (num_neu_bias, num_neu_res + num_neu_out)) * in_scale * 0
W_fb = np.random.uniform(-1, 1, (num_neu_res, num_neu_out)) > 0

W_in_quantized = W_in.tolist()
W_res_quantized = W_res.tolist()
W_out_quantized = W_out.tolist()
W_bias_quantized = W_bias.tolist()
W_fb = W_fb.astype(bool).tolist()

# Log-normal distribution parameters
mu, sigma = 0.5, 0.5
num_samples = num_neu_res

# Generate initial log-normal samples
tau_samples = np.random.lognormal(mu, sigma, num_samples)

# Define the boundaries for resampling
lower_bound = 0.0
upper_bound = 6.0

# Separate samples into bins
inside_bounds = (tau_samples >= lower_bound) & (tau_samples <= upper_bound)
tau_samples_resampled = tau_samples[inside_bounds]

while np.sum(~inside_bounds) > 0:
    resampled_values = np.random.lognormal(mu, sigma, np.sum(~inside_bounds))
    inside_bounds_new = (resampled_values >= lower_bound) & (resampled_values <= upper_bound)
    tau_samples_resampled = np.concatenate((tau_samples_resampled, resampled_values[inside_bounds_new]))
    inside_bounds = inside_bounds_new

tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56]) * 2500
bins = np.linspace(0.0, 6.0, len(tau_values) + 1)
tau_bins = np.digitize(tau_samples_resampled, bins)
tau_bins = np.clip(tau_bins - 1, 0, len(tau_values) - 1)
tau_array_hetero = tau_values[tau_bins]

# Results validation
mean_actual = np.mean(tau_array_hetero)
print(f"Actual mean: {mean_actual}")

unique, counts = np.unique(tau_array_hetero, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Value {value}: {count} samples")

tau_homo_value = 20000.0
tau_array_homo = np.full(num_neu_res, tau_homo_value)
print(tau_homo_value)

tau_type = "hetero" # or "homo"

if tau_type == "hetero":
    tau_array = tau_array_hetero
else:
    tau_array = tau_array_homo

# Define the core parameters dictionary
core_parameters = {
    "t_delay": 3,
    "V_init": 0.0,
    "tau_out": 10000.0,
    "V_bot": -1.0,
    "V_th": 1.0,
    "V_reset": 0.0,
    "t_ref": 4,
    "SG_window": 0.5,
    "N_in": num_neu_in,
    "N_res": num_neu_res,
    "N_out": num_neu_out,
    "N_bias": num_neu_bias,
    "N_class": num_class,
    "N_out_times": num_out_times,
    "PTE_slide": 0,
    "PTE_times": 4,
    "PTE_range": 1,
    "ET_N": 15,
}

# Define the system parameters dictionary
system_parameters = {
    "T_sim": 1000000000,
    "epoch": 100,                                               # Example epoch value
    "lr": 0.004,                                                # same as conductance steps. This is for 8bits ~ 1/250.
    "test_file": "../speech2spikes/tools/gen_spike/test.bin",    # Replace with the actual test file path
    "training_file": "../speech2spikes/tools/gen_spike/train",   # Replace with the actual training file path
    # /home/sungminlee/Speakmin_draft/SpeakMin/speech2spikes/tools/gen_spike/train0.bin
    # /home/sungminlee/Speakmin_draft/SpeakMin/speech2spikes/tools/gen_spike/final_new_num_dataset2
    "N_chunks": 10,                             # you can devide training dataset as 'chunk'
}

# Combine system and core parameters into a single dictionary
parameters = {
    "system_parameter": system_parameters,
    "core_parameter": core_parameters
}

# Define the weights dictionary with quantized weights
weights = {
    "W_in": W_in_quantized,
    "W_res": W_res_quantized,
    "W_out": W_out_quantized,
    "W_bias": W_bias_quantized,
    "W_fb": W_fb
}

# Paths to the JSON files in the src directory
parameters_path = './src/init_parameters.json'
weights_path = './src/init_weights.json'
tau_path = './src/init_taus.json'

# Save the parameters to the init_parameters.json file
with open(parameters_path, 'w') as f:
    json.dump(parameters, f, indent=4)
print(f"Parameters file saved to {parameters_path}")

# Save the weights to the init_weights.json file
with open(weights_path, 'w') as f:
    json.dump(weights, f, indent=4)
print(f"Weights file saved to {weights_path}")

# Save the tau values to the init_taus.json file
tau_dict = {"tau": tau_array.tolist()}
with open(tau_path, 'w') as f:
    json.dump(tau_dict, f, indent=4)
print(f"Tau file saved to {tau_path}")

# Check the distribution of tau values
tau_distribution = {tau: list(tau_array).count(tau) for tau in tau_values}
print(f"Tau distribution: {tau_distribution}")

# Print only the variables from the parameters file
print("\nParameters Variables:")
for key, value in parameters.items():
    print(f"{key}: {value}")