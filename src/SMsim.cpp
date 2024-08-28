// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#include <iomanip>
#include <getopt.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <bitset>

#include "Core.h"

#ifndef __GIT_REV__
#define __GIT_REV__ "unknown"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// Byte reversal functions
uint32_t reverse_bytes(uint32_t value) {
    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}

uint16_t reverse_bytes(uint16_t value) {
    return ((value & 0x00FF) << 8) |
           ((value & 0xFF00) >> 8);
}

// Calculate offsets for each entry in the binary file
std::vector<std::streampos> calculate_offsets(const std::string& file_path, int& num_entries) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open binary file");
    }

    std::vector<std::streampos> offsets;
    uint8_t label;
    uint16_t data_index;
    uint32_t unique_global_id;
    uint32_t num_data_points;
    uint8_t reserved1;
    uint8_t reserved2;

    const int HEADER_SIZE = 13; // bytes

    while (file.read(reinterpret_cast<char*>(&label), sizeof(uint8_t))) {
        file.read(reinterpret_cast<char*>(&data_index), sizeof(uint16_t));
        data_index = reverse_bytes(data_index);

        file.read(reinterpret_cast<char*>(&unique_global_id), sizeof(uint32_t));
        unique_global_id = reverse_bytes(unique_global_id);

        file.read(reinterpret_cast<char*>(&num_data_points), sizeof(uint32_t));
        num_data_points = reverse_bytes(num_data_points);

        file.read(reinterpret_cast<char*>(&reserved1), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&reserved2), sizeof(uint8_t));

        std::streampos current_pos = file.tellg();
        offsets.push_back(current_pos - std::streamoff(HEADER_SIZE));

        file.seekg(num_data_points * (sizeof(uint32_t) + sizeof(uint16_t)), std::ios::cur);
    }

    num_entries = offsets.size();
    file.close();
    return offsets;
}

// Load spike trains in parallel
void load_spike_trains_parallel(const std::string& file_path, std::vector<std::vector<uint32_t>>& all_spike_times, std::vector<std::vector<uint16_t>>& all_neuron_indices, std::vector<uint8_t>& all_labels, const std::vector<std::streampos>& offsets) {
    int num_entries = offsets.size();
    all_spike_times.resize(num_entries);
    all_neuron_indices.resize(num_entries);
    all_labels.resize(num_entries);

    #pragma omp parallel for
    for (int i = 0; i < num_entries; ++i) {
        std::ifstream local_file(file_path, std::ios::binary);
        if (!local_file.is_open()) {
            throw std::runtime_error("Could not open binary file");
        }

        local_file.seekg(offsets[i]);

        uint8_t label;
        uint16_t data_index;
        uint32_t unique_global_id;
        uint32_t num_data_points;
        uint8_t reserved1;
        uint8_t reserved2;

        local_file.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));

        local_file.read(reinterpret_cast<char*>(&data_index), sizeof(uint16_t));
        data_index = reverse_bytes(data_index);

        local_file.read(reinterpret_cast<char*>(&unique_global_id), sizeof(uint32_t));
        unique_global_id = reverse_bytes(unique_global_id);

        local_file.read(reinterpret_cast<char*>(&num_data_points), sizeof(uint32_t));
        num_data_points = reverse_bytes(num_data_points);

        local_file.read(reinterpret_cast<char*>(&reserved1), sizeof(uint8_t));
        local_file.read(reinterpret_cast<char*>(&reserved2), sizeof(uint8_t));

        std::vector<uint32_t> spike_times(num_data_points);
        std::vector<uint16_t> neuron_indices(num_data_points);

        for (uint32_t j = 0; j < num_data_points; ++j) {
            local_file.read(reinterpret_cast<char*>(&spike_times[j]), sizeof(uint32_t));
            spike_times[j] = reverse_bytes(spike_times[j]);

            local_file.read(reinterpret_cast<char*>(&neuron_indices[j]), sizeof(uint16_t));
            neuron_indices[j] = reverse_bytes(neuron_indices[j]);
        }

        all_spike_times[i] = std::move(spike_times);
        all_neuron_indices[i] = std::move(neuron_indices);

        // you could change this part for another classes
        all_labels[i] = label;
        // all_labels[i] = label-10;

        local_file.close();
    }

    // Print all information for the first data entry
    if (!all_spike_times.empty() && !all_neuron_indices.empty()) {
        uint8_t label;
        uint16_t data_index;
        uint32_t unique_global_id;
        uint32_t num_data_points;
        uint8_t reserved1;
        uint8_t reserved2;

        std::ifstream local_file(file_path, std::ios::binary);
        if (!local_file.is_open()) {
            throw std::runtime_error("Could not open binary file");
        }

        local_file.seekg(offsets[0]);
        local_file.read(reinterpret_cast<char*>(&label), sizeof(uint8_t));
        local_file.read(reinterpret_cast<char*>(&data_index), sizeof(uint16_t));
        data_index = reverse_bytes(data_index);

        local_file.read(reinterpret_cast<char*>(&unique_global_id), sizeof(uint32_t));
        unique_global_id = reverse_bytes(unique_global_id);

        local_file.read(reinterpret_cast<char*>(&num_data_points), sizeof(uint32_t));
        num_data_points = reverse_bytes(num_data_points);

        local_file.read(reinterpret_cast<char*>(&reserved1), sizeof(uint8_t));
        local_file.read(reinterpret_cast<char*>(&reserved2), sizeof(uint8_t));

        /*
        std::cout << "Label: " << static_cast<int>(label) << " (" << std::bitset<8>(label) << ")\n";
        std::cout << "Data Index: " << data_index << " (" << std::bitset<16>(data_index) << ")\n";
        std::cout << "Unique Global ID: " << unique_global_id << " (" << std::bitset<32>(unique_global_id) << ")\n";
        std::cout << "Num Data Points: " << num_data_points << " (" << std::bitset<32>(num_data_points) << ")\n";
        std::cout << "Reserved1: " << static_cast<int>(reserved1) << " (" << std::bitset<8>(reserved1) << ")\n";
        std::cout << "Reserved2: " << static_cast<int>(reserved2) << " (" << std::bitset<8>(reserved2) << ")\n";

        // Print the first 10 spike times and neuron indices
        std::cout << "First 10 spike times and neuron indices of the first entry:" << std::endl;
        for (size_t j = 0; j < std::min(size_t(10), all_spike_times[0].size()); ++j) {
            std::cout << "Spike Time: " << all_spike_times[0][j] << " (binary: " << std::bitset<32>(all_spike_times[0][j]) << "), Neuron Index: " << all_neuron_indices[0][j] << " (binary: " << std::bitset<16>(all_neuron_indices[0][j]) << ")" << std::endl;
        }

        // Print the last 10 spike times and neuron indices
        std::cout << "Last 10 spike times and neuron indices of the first entry:" << std::endl;
        for (size_t j = std::max(all_spike_times[0].size(), size_t(10)) - 10; j < all_spike_times[0].size(); ++j) {
            std::cout << "Spike Time: " << all_spike_times[0][j] << " (binary: " << std::bitset<32>(all_spike_times[0][j]) << "), Neuron Index: " << all_neuron_indices[0][j] << " (binary: " << std::bitset<16>(all_neuron_indices[0][j]) << ")" << std::endl;
        }
        */
    }
}

// Format the duration into hours, minutes, and seconds
std::string format_duration(std::chrono::duration<double> duration) {
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(seconds);
    auto hours = std::chrono::duration_cast<std::chrono::hours>(minutes);
    seconds -= std::chrono::duration_cast<std::chrono::seconds>(minutes);
    minutes -= std::chrono::duration_cast<std::chrono::minutes>(hours);

    std::ostringstream oss;
    if (hours.count() > 0) {
        oss << hours.count() << "h ";
    }
    if (minutes.count() > 0 || hours.count() > 0) {
        oss << minutes.count() << "m ";
    }
    oss << seconds.count() << "s";
    return oss.str();
}

// Print the progress bar for current processing
void print_progress_bar(int current, int total, const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) {
    int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = bar_width * progress;

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << format_duration(elapsed) << " elapsed)\r";
    std::cout.flush();
}

// Run the simulation and return the accuracy
double run_simulation(Core& core_template, const std::string& file_path, int epoch, const std::string& type, int& data_count) {
    int correct_count = 0;
    bool enabling_train = (type == "train");

    std::vector<std::vector<uint32_t>> all_spike_times;
    std::vector<std::vector<uint16_t>> all_neuron_indices;
    std::vector<uint8_t> all_labels;

    int num_entries;
    std::vector<std::streampos> offsets = calculate_offsets(file_path, num_entries);

    load_spike_trains_parallel(file_path, all_spike_times, all_neuron_indices, all_labels, offsets);

    data_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Current learning rate is " << core_template.lr << std::endl;

    for (size_t i = 0; i < all_spike_times.size(); ++i) {
        if ((type == "train" && data_count >= 10000) || (type == "test" && data_count >= 1000)) break;

        core_template.reset(); // Reset neurons and spike queues
        core_template.enabling_train = enabling_train;
        core_template.load_spike_train(all_spike_times[i], all_neuron_indices[i]);
        core_template.class_label = all_labels[i];

        bool is_correct = core_template.run();

        if (is_correct) {
            ++correct_count;
        }
        ++data_count;

        print_progress_bar(data_count, (type == "train") ? 10000 : 1000, start_time);

        if (data_count % 1000 == 0) {
            double current_accuracy = static_cast<double>(correct_count) / data_count;
            std::cout << "Current accuracy after " << data_count << " data points: " << current_accuracy * 100 << "%" << std::endl;
        }
    }

    std::cout << std::endl;
    return static_cast<double>(correct_count) / data_count;
}

// Print progress bar for epochs
void print_epoch_progress(int epoch, int total_epochs, const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) {
    int bar_width = 50;
    float progress = static_cast<float>(epoch) / total_epochs;
    int pos = bar_width * progress;

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << format_duration(elapsed) << " elapsed)\r";
    std::cout.flush();
}

// Save accuracy to file
void save_accuracy_to_file(const std::string& filename, int epoch, double train_accuracy, double test_accuracy) {
    std::ofstream file;
    file.open(filename, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open accuracy file");
    }

    file << epoch << "," << train_accuracy << "," << test_accuracy << "\n";
    file.close();
}

int main(int argc, char *argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting SMsim..." << std::endl;

    // File paths
    std::string param_file = "./init_parameters.json";
    std::string weights_file = "./init_weights.json";
    std::string tau_file = "./init_taus.json";

    // Load system parameters from JSON
    std::ifstream param_ifs(param_file);
    if (!param_ifs.is_open()) {
        throw std::runtime_error("Could not open parameter file");
    }
    json param_json;
    param_ifs >> param_json;

    int num_epochs = param_json["system_parameter"]["epoch"].get<int>();
    std::string base_train_file_path = param_json["system_parameter"]["training_file"].get<std::string>();
    std::string test_file_path = param_json["system_parameter"]["test_file"].get<std::string>();
    int T_sim = param_json["system_parameter"]["T_sim"].get<int>();
    double lr = param_json["system_parameter"]["lr"].get<double>();
    int N_chunks = param_json["system_parameter"]["N_chunks"].get<int>();

    std::cout << "Loaded system parameters:" << std::endl;
    std::cout << "Epochs: " << num_epochs << std::endl;
    std::cout << "Training file path: " << base_train_file_path << std::endl;
    std::cout << "Test file path: " << test_file_path << std::endl;
    std::cout << "Simulation time (T_sim): " << T_sim << std::endl;

    std::cout << "version: " <<  __GIT_REV__ << std::endl;
    std::cout << "CXX: " <<  __VERSION__ << std::endl;

    // read tau file
    std::ifstream tau_ifs(tau_file);
    if (!tau_ifs.is_open()) {
        throw std::runtime_error("Could not open tau file");
    }
    json tau_json;
    tau_ifs >> tau_json;
    std::vector<int> tau_values = tau_json["tau"].get<std::vector<int>>();

    // generate accuracy file
    const std::string accuracy_file = "accuracy_log.csv";
    std::ofstream file(accuracy_file);
    if (file.is_open()) {
        file << "epoch, train_accuracy, test_accuracy \n";
        file.close();
    }

    // initialization of the core
    Core core_template(param_file, weights_file, tau_values);
    core_template.T_sim = T_sim;
    core_template.lr = lr;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        if (epoch > 0) {
            if (fs::exists("./training_weights.json")) {
                core_template.load_weights("./training_weights.json");
                std::cout << "Loaded weights for epoch " << epoch << std::endl;
            }
        }

        print_epoch_progress(epoch, num_epochs, program_start);

        std::cout << "\nStarting training epoch " << epoch << "...\n";

        int chunk_index = epoch % N_chunks;
        std::cout << chunk_index << "...\n";
#if defined(TRAIN_PHASE)
        core_template.PTE_slide = (epoch / N_chunks) % core_template.PTE_times;
#endif
        std::stringstream ss;
        ss << base_train_file_path << chunk_index << ".bin";
        std::string train_file_path = ss.str();

        int train_data_count;
        int test_data_count;

        double train_result = run_simulation(core_template, train_file_path, epoch, "train", train_data_count);
        std::cout << "Epoch " << epoch << " training accuracy: " << train_result * 100 << "%" << " with " << train_data_count << " data points." << std::endl;

        if (epoch % 5 == 0) {
            std::cout << "Starting testing epoch " << epoch << "...\n";

            double test_result = run_simulation(core_template, test_file_path, epoch, "test", test_data_count);
            std::cout << "Epoch " << epoch << " test accuracy: " << test_result * 100 << "%" << " with " << test_data_count << " data points." << std::endl;
            auto epoch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
            std::cout << "Epoch " << epoch << " duration: " << format_duration(epoch_duration) << ".\n";

            save_accuracy_to_file(accuracy_file, epoch, train_result * 100, test_result * 100);
        }

        core_template.save_weights("./training_weights.json");
    }

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "User time " << usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 << " sec" << std::endl;
    std::cout << "System time " << usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6 << " sec" << std::endl;
    std::cout << "maxurss " << usage.ru_maxrss / 1024 << " MB" << std::endl;

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> program_duration = program_end - program_start;
    std::cout << "Total program duration: " << format_duration(program_duration) << ".\n";

    return 0;
}

// Load a single spike train for verification
void load_single_spike_train(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open binary file");
    }

    uint8_t label;
    uint16_t data_index;
    uint32_t unique_global_id;
    uint32_t num_data_points;
    uint8_t reserved1;
    uint8_t reserved2;

    // Read header
    file.read(reinterpret_cast<char*>(&label), sizeof(label));
    file.read(reinterpret_cast<char*>(&data_index), sizeof(data_index));
    data_index = reverse_bytes(data_index);

    file.read(reinterpret_cast<char*>(&unique_global_id), sizeof(unique_global_id));
    unique_global_id = reverse_bytes(unique_global_id);

    file.read(reinterpret_cast<char*>(&num_data_points), sizeof(num_data_points));
    num_data_points = reverse_bytes(num_data_points);

    file.read(reinterpret_cast<char*>(&reserved1), sizeof(reserved1));
    file.read(reinterpret_cast<char*>(&reserved2), sizeof(reserved2));

    std::vector<uint32_t> spike_times(num_data_points);
    std::vector<uint16_t> neuron_indices(num_data_points);

    // Read spike times and neuron indices
    for (uint32_t j = 0; j < num_data_points; ++j) {
        file.read(reinterpret_cast<char*>(&spike_times[j]), sizeof(uint32_t));
        spike_times[j] = reverse_bytes(spike_times[j]);

        file.read(reinterpret_cast<char*>(&neuron_indices[j]), sizeof(uint16_t));
        neuron_indices[j] = reverse_bytes(neuron_indices[j]);
    }

    // Close the file
    file.close();

    /*
    // Print the header and first/last 10 spikes for verification
    std::cout << "Label: " << static_cast<int>(label) << " (" << std::bitset<8>(label) << ")\n";
    std::cout << "Data Index: " << data_index << " (" << std::bitset<16>(data_index) << ")\n";
    std::cout << "Unique Global ID: " << unique_global_id << " (" << std::bitset<32>(unique_global_id) << ")\n";
    std::cout << "Num Data Points: " << num_data_points << " (" << std::bitset<32>(num_data_points) << ")\n";
    std::cout << "Reserved1: " << static_cast<int>(reserved1) << " (" << std::bitset<8>(reserved1) << ")\n";
    std::cout << "Reserved2: " << static_cast<int>(reserved2) << " (" << std::bitset<8>(reserved2) << ")\n";

    std::cout << "Spike Times and Neuron Indices (First 10 entries):\n";
    for (size_t j = 0; j < std::min(size_t(10), spike_times.size()); ++j) {
        std::cout << "    Spike Time: " << spike_times[j] << " (" << std::bitset<32>(spike_times[j]) << "), Neuron Index: " << neuron_indices[j] << " (" << std::bitset<16>(neuron_indices[j]) << ")\n";
    }

    std::cout << "Spike Times and Neuron Indices (Last 10 entries):\n";
    for (size_t j = spike_times.size() > 10 ? spike_times.size() - 10 : 0; j < spike_times.size(); ++j) {
        std::cout << "    Spike Time: " << spike_times[j] << " (" << std::bitset<32>(spike_times[j]) << "), Neuron Index: " << neuron_indices[j] << " (" << std::bitset<16>(neuron_indices[j]) << ")\n";
    }

    // Check if the number of 1s in neuron indices matches num_data_points
    int total_ones = 0;
    for (const auto& index : neuron_indices) {
        total_ones += std::bitset<16>(index).count();
    }

    std::cout << "Total number of ones: " << total_ones << "\n";
    if (total_ones == num_data_points) {
        std::cout << "Total number of ones matches num_data_points.\n";
    } else {
        std::cout << "Total number of ones (" << total_ones << ") does not match num_data_points (" << num_data_points << ").\n";
    }
    */
}
