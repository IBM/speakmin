// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#include "Core.h"
#include "Config.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <vector>
#include <thread>

using json = nlohmann::json;

// Core constructor with distributed tau values
Core::Core(const std::string& param_file, const std::string& weights_file, const std::vector<int>& tau_values) {

    std::cout << "Parameter file path: " << param_file << std::endl;
    std::cout << "Weights file path: " << weights_file << std::endl;

    Config config;

    // Read and parse parameter file
    std::ifstream param_stream(param_file);
    if (!param_stream.is_open()) {
        throw std::runtime_error("Could not open parameter file");
    }
    json param_json;
    param_stream >> param_json;

    config.t_delay = param_json["core_parameter"]["t_delay"].get<uint32_t>();
    config.V_init = param_json["core_parameter"]["V_init"].get<double>();
    config.tau_out = param_json["core_parameter"]["tau_out"].get<double>();
    config.V_th = param_json["core_parameter"]["V_th"].get<double>();
    config.V_bot = param_json["core_parameter"]["V_bot"].get<double>();
    config.V_reset = param_json["core_parameter"]["V_reset"].get<double>();
    config.SG_window = param_json["core_parameter"]["SG_window"].get<double>();
#if defined(REFRACTORY)
    config.t_ref = param_json["core_parameter"]["t_ref"].get<uint32_t>();
#endif
    config.N_in = param_json["core_parameter"]["N_in"].get<int>();
    config.N_res = param_json["core_parameter"]["N_res"].get<int>();
    config.N_out = param_json["core_parameter"]["N_out"].get<int>();
    config.N_bias = param_json["core_parameter"]["N_bias"].get<int>();
    config.N_class = param_json["core_parameter"]["N_class"].get<int>();
    config.N_out_times = param_json["core_parameter"]["N_out_times"].get<int>();

    config.PTE_slide = param_json["core_parameter"]["PTE_slide"].get<uint32_t>();
    config.PTE_times = param_json["core_parameter"]["PTE_times"].get<uint32_t>();
    config.PTE_range = param_json["core_parameter"]["PTE_range"].get<uint32_t>();
    config.ET_N = param_json["core_parameter"]["ET_N"].get<uint32_t>();

    /*
    // Print the loaded values
    std::cout << "t_delay: " << config.t_delay << std::endl;
    std::cout << "V_init: " << config.V_init << std::endl;
    std::cout << "tau_out: " << config.tau_out << std::endl;
    std::cout << "V_th: " << config.V_th << std::endl;
    std::cout << "V_bot: " << config.V_bot << std::endl;
    std::cout << "V_reset: " << config.V_reset << std::endl;
    std::cout << "SG_window: " << config.SG_window << std::endl;
    #if defined(REFRACTORY)
    std::cout << "t_ref: " << config.t_ref << std::endl;
    #endif
    std::cout << "N_in: " << config.N_in << std::endl;
    std::cout << "N_res: " << config.N_res << std::endl;
    std::cout << "N_out: " << config.N_out << std::endl;
    std::cout << "N_bias: " << config.N_bias << std::endl;
    std::cout << "N_class: " << config.N_class << std::endl;
    std::cout << "N_out_times: " << config.N_out_times << std::endl;

    std::cout << "PTE_slide: " << config.PTE_slide << std::endl;
    std::cout << "PTE_times: " << config.PTE_times << std::endl;
    std::cout << "PTE_range: " << config.PTE_range << std::endl;
    std::cout << "ET_N: " << config.ET_N << std::endl;
    */

    std::cout << "parameter file is okay" << std::endl;

    // Read and parse weights file
    std::ifstream weights_stream(weights_file);
    if (!weights_stream.is_open()) {
        throw std::runtime_error("Could not open weights file");
    }
    json weights_json;
    weights_stream >> weights_json;

    config.W_in = weights_json["W_in"].get<std::vector<std::vector<double>>>();
    config.W_res = weights_json["W_res"].get<std::vector<std::vector<double>>>();
    config.W_out = weights_json["W_out"].get<std::vector<std::vector<double>>>();
    config.W_fb = weights_json["W_fb"].get<std::vector<std::vector<bool>>>();
    config.W_bias = weights_json["W_bias"].get<std::vector<std::vector<double>>>();

    std::cout << "weights file is okay" << std::endl;

    *this = Core(config, tau_values);
}


// Core constructor with configuration and tau values
Core::Core(const Config& config, const std::vector<int>& tau_values)
    : W_in(config.W_in), W_res(config.W_res), W_out(config.W_out), W_fb(config.W_fb), W_bias(config.W_bias), T_sim(config.T_sim), t_delay(config.t_delay) {

#if defined(REFRACTORY)
    Neu_res.reserve(config.N_res);
    for (int i = 0; i < config.N_res; ++i) {
        Neu_res.emplace_back(config.V_init, tau_values[i], config.V_th, config.V_bot, config.V_reset, config.t_ref, config.SG_window);
    }

    Neu_out.reserve(config.N_out);
    for (int i = 0; i < config.N_out; ++i) {
        Neu_out.emplace_back(config.V_init, config.tau_out, config.V_th, config.V_bot, config.V_reset, config.t_ref, config.SG_window);
    }
#else
    Neu_res.reserve(config.N_res);
    for (int i = 0; i < config.N_res; ++i) {
        Neu_res.emplace_back(config.V_init, tau_values[i], config.V_th, config.V_bot, config.V_reset, config.SG_window);
    }

    Neu_out.reserve(config.N_out);
    for (int i = 0; i < config.N_out; ++i) {
        Neu_out.emplace_back(config.V_init, config.tau_out, config.V_th, config.V_bot, config.V_reset, config.SG_window);
    }
#endif

    Neu_acc.resize(config.N_class, 0);
    N_out_times = config.N_out_times;

    PTE_slide = config.PTE_slide;
    PTE_times = config.PTE_times;
    PTE_range = config.PTE_range;
    ET_N = config.ET_N;

    /*
    // Check the initialized states of Neu_res, Neu_out
    std::cout << "Neu_res size: " << Neu_res.size() << std::endl;
    std::cout << "Neu_out size: " << Neu_out.size() << std::endl;

    // Check the initialized states of Neu_acc
    std::cout << "Neu_acc size: " << Neu_acc.size() << std::endl;
    for (size_t i = 0; i < Neu_acc.size(); ++i) {
        std::cout << "Neu_acc[" << i << "] = " << Neu_acc[i] << std::endl;
    }
    */

    /**/
    std::cout << "N_out_times: " << N_out_times << std::endl;
    std::cout << "PTE_slide: " << PTE_slide << std::endl;
    std::cout << "PTE_times: " << PTE_times<< std::endl;
    std::cout << "PTE_range: " << PTE_range << std::endl;
    std::cout << "ET_N: " << ET_N<< std::endl;

}

// Copy constructor
Core::Core(const Core& other)
    : W_in(other.W_in), W_res(other.W_res), W_out(other.W_out), W_fb(other.W_fb), W_bias(other.W_bias), T_sim(other.T_sim), t_delay(other.t_delay), Neu_res(other.Neu_res), Neu_out(other.Neu_out), Neu_acc(other.Neu_acc), external_S_queue(other.external_S_queue), internal_S_queue(other.internal_S_queue), S_vec_now(other.S_vec_now), N_out_times(other.N_out_times), enabling_train(other.enabling_train), class_label(other.class_label), lr(other.lr) {
}

// Assignment operator
Core& Core::operator=(const Core& other) {
    if (this != &other) {
        W_in = other.W_in;
        W_res = other.W_res;
        W_out = other.W_out;
        W_fb = other.W_fb;
        W_bias = other.W_bias;
        T_sim = other.T_sim;
        t_delay = other.t_delay;
        Neu_res = other.Neu_res;
        Neu_out = other.Neu_out;
        Neu_bias = other.Neu_bias;
        Neu_acc = other.Neu_acc;
        external_S_queue = other.external_S_queue;
        internal_S_queue = other.internal_S_queue;
        S_vec_now = other.S_vec_now;
        N_out_times = other.N_out_times;
        enabling_train = other.enabling_train;
        class_label = other.class_label;
        lr = other.lr;
    }
    return *this;
}

// Reset the core
void Core::reset() {
    /*
    for (auto& neuron : Neu_res) {
#if defined(REFRACTORY)
        neuron = Neuron(neuron.get_V_mem(), neuron.get_tau(), neuron.get_V_th(), neuron.get_V_bot(), neuron.get_V_reset(), neuron.get_t_ref(), neuron.get_SG_window());
#else
        neuron = Neuron(neuron.get_V_mem(), neuron.get_tau(), neuron.get_V_th(), neuron.get_V_bot(), neuron.get_V_reset(), neuron.get_SG_window());
#endif
        }

    for (auto& neuron : Neu_out) {
#if defined(REFRACTORY)
        neuron = Neuron(neuron.get_V_mem(), neuron.get_tau(), neuron.get_V_th(), neuron.get_V_bot(), neuron.get_V_reset(), neuron.get_t_ref(), neuron.get_SG_window());
#else
        neuron = Neuron(neuron.get_V_mem(), neuron.get_tau(), neuron.get_V_th(), neuron.get_V_bot(), neuron.get_V_reset(), neuron.get_SG_window());
#endif
    }*/

    /**/
    for (size_t i = 0; i < Neu_res.size(); ++i) {
        Neu_res[i].reset();
        Neu_res[i].reset_ref();
    }

    for (size_t i = 0; i < Neu_out.size(); ++i) {
        Neu_out[i].reset();
        Neu_out[i].reset_ref();
    }


    while (!external_S_queue.empty()) {
        external_S_queue.pop();
    }

    while (!internal_S_queue.empty()) {
        internal_S_queue.pop();
    }

    while (!Event_queue.empty()) {
        Event_queue.pop();
    }

    while (!Event_queue_delay.empty()) {
        Event_queue_delay.pop();
    }

    while (!S_vec_trace.empty()) {
        S_vec_trace.pop();
    }

    while (!S_vec_trace_delay.empty()) {
        S_vec_trace_delay.pop();
    }

    S_vec_now.clear();
    Neu_acc.assign(Neu_acc.size(), 0);  // Reset the size of Neu_acc based on the current number of classes
}

// Save recorded spikes to a file
void Core::save_recorded_spikes(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open binary file for saving recorded spikes");
    }

    for (size_t i = 0; i < recorded_times.size(); ++i) {
        file.write(reinterpret_cast<const char*>(&recorded_times[i]), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&recorded_neuron_indices[i]), sizeof(uint16_t));
    }

    file.close();
}

// Run the simulation
bool Core::run() {
    return run_loop();
}

// Run the simulation loop
bool Core::run_loop() {

    omp_set_num_threads(4);

    uint32_t T_now = 0;
    size_t class_now = static_cast<size_t>(class_label);
    size_t train_signal;

    while (T_now <= T_sim) {
        // if (internal_S_queue.empty()) break;
        uint32_t T_external = external_S_queue.empty() ? T_sim + 1 : external_S_queue.top().time;
        uint32_t T_internal = internal_S_queue.empty() ? T_sim + 1 : internal_S_queue.top().time;
        T_now = std::min(T_external, T_internal);

        // std::cout << T_now << std::endl;

        if (T_now > T_sim) break;

        while (!external_S_queue.empty() && external_S_queue.top().time <= T_now) {
            S_vec_now.push_back(external_S_queue.top());
            external_S_queue.pop();
        }

        while (!internal_S_queue.empty() && internal_S_queue.top().time <= T_now) {
            S_vec_now.push_back(internal_S_queue.top());
            internal_S_queue.pop();
        }

        if (N_out_times == 0) {
            throw std::runtime_error("N_out_times cannot be zero");
        }

        train_index = static_cast<size_t>(static_cast<size_t>(T_now) % N_out_times);
        train_signal = 0;

#if defined(TRAIN_PHASE)
        if (PTE_times == 0) {
            throw std::runtime_error("N_training_times cannot be zero");
        }
        if (PTE_range == 0) {
            throw std::runtime_error("PTE_range cannot be zero");
        }

        // std::cout << "PTE_slide: " << PTE_slide << ", PTE_times: " << PTE_times << ", PTE_range: " << PTE_range << std::endl;
        train_signal = static_cast<size_t>(((static_cast<size_t>(T_now) + PTE_range * PTE_slide) / PTE_range) % PTE_times);
#endif

        // Parallelize the leak method for Neu_res and Neu_out
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < Neu_res.size(); ++i) {
                Neu_res[i].leak(T_now);
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < Neu_out.size(); ++i) {
                Neu_out[i].leak(T_now);
            }
        }

        // std::cout << "leaky is OKAY " << std::endl;

        // Parallelize spike propagation
        #pragma omp parallel for
        for (size_t i = 0; i < S_vec_now.size(); ++i) {
            const auto& S_now = S_vec_now[i];
            int id_now = S_now.id.first;
            char layer = S_now.id.second;

            if (layer == 'r') {
                for (size_t j = 0; j < Neu_res.size(); ++j) {
                    Neu_res[j].in(W_res[id_now][j]);
                }
                for (size_t j = 0; j < Neu_out.size(); ++j) {
                    Neu_out[j].in(W_out[id_now][j]);
                }
            } else if (layer == 'i') {
                for (size_t j = 0; j < Neu_res.size(); ++j) {
                    Neu_res[j].in(W_in[id_now][j]);
                }
            }
        }
#if defined(TRAIN_FA) || defined(TRAIN_DFA)
        while (!Event_queue_delay.empty() && Event_queue_delay.top().time <= T_now) {
            #pragma omp critical
            {
                Event_vec_now.push_back(Event_queue_delay.top());
                Event_queue_delay.pop();
            }
        }
#endif
#if defined(TRAIN_ELIGIBLETRACE)
        while (!S_vec_trace.empty() && S_vec_trace.top().time <= T_now) {
            #pragma omp critical
            {
            S_vec_trace_now.push_back(S_vec_trace.top());
            S_vec_trace.pop();
            }
        }
#endif
        // std::cout << "after spike sampling " << std::endl;

        // checking firing
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                #pragma omp parallel for
                for (size_t i = 0; i < Neu_res.size(); ++i) {
                    if (Neu_res[i].is_firing()) {
                        if (enabling_train) {
#if defined(TRAIN_FA) || defined(TRAIN_DFA)
                            bool SG_now = Neu_res[i].get_SG();
                            if (SG_now) {
                                for (const auto& S_now : S_vec_now) {
                                    int id_now = S_now.id.first;
                                    char layer = S_now.id.second;
                                    if (layer == 'r') {
                                        std::pair<int, char> spk_id = std::make_pair(id_now, 'r');
                                        std::pair<int, char> neu_id = std::make_pair(i, 'r');
                                        Event_unit event(T_now + t_delay, spk_id, neu_id, true);
                                        #pragma omp critical
                                        {
                                            Event_queue_delay.push(event);
                                        }
                                    }
                                    /*
                                    else if (layer == 'i') {
                                        std::pair<int, char> spk_id = std::make_pair(id_now, 'i');
                                        std::pair<int, char> neu_id = std::make_pair(i, 'r');
                                        Event_unit event(T_now + t_delay, spk_id, neu_id, true);
                                        #pragma omp critical
                                        {
                                            Event_queue_delay.push(event);
                                        }
                                    }
                                    */
                                }
                            }
#endif
#if defined(TRAIN_ELIGIBLETRACE)
                            #pragma omp critical
                            {

                                for (int n = 0; n < ET_N; ++n) {
                                    S_vec_trace.push(Spike(T_now + t_delay + n + 1, {i, 'r'}));
                                }
                            }
#endif
                        }
                        #pragma omp critical
                        {
                            internal_S_queue.push(Spike(T_now + t_delay, {i, 'r'}));
                        }
                        Neu_res[i].reset();
                    }
                }
            }

            #pragma omp section
            {
                #pragma omp parallel for
                for (size_t i = 0; i < Neu_out.size(); ++i) {
                    if (Neu_out[i].is_firing()) {
                        bool SG_now = Neu_out[i].get_SG();
#if defined(TRAIN_PHASE)
                        if (enabling_train && !train_signal)
#else
                        if (enabling_train)
#endif
                        {
                            if ( (static_cast<size_t>(i / N_out_times) != class_now) && (static_cast<size_t>(i % N_out_times) == train_index) ) {
                                if (SG_now) {
                                    for (const auto& S_now : S_vec_now) {
                                        int id_now = S_now.id.first;
                                        char layer = S_now.id.second;

                                        if (layer == 'r') {
                                            std::pair<int, char> spk_id = std::make_pair(id_now, 'r');
                                            std::pair<int, char> neu_id = std::make_pair(i, 'o');
                                            Event_unit event(T_now, spk_id, neu_id, false);
                                            #pragma omp critical
                                            {
                                                Event_queue.push(event);
                                            }
                                        }
                                    }
    #if defined(TRAIN_ELIGIBLETRACE)
                                    for (const auto& S_now : S_vec_trace_now) {
                                        int id_now = S_now.id.first;
                                        char layer = S_now.id.second;

                                        if (layer == 'r') {
                                            std::pair<int, char> spk_id = std::make_pair(id_now, 'r');
                                            std::pair<int, char> neu_id = std::make_pair(i, 'o');
                                            Event_unit event(T_now, spk_id, neu_id, false);
                                            #pragma omp critical
                                            {
                                                Event_queue.push(event);
                                            }
                                        }
                                    }
    #endif
    #if defined(TRAIN_FA)
                                    for (const auto& E_now : Event_vec_now) {
                                    int spk_id_now = E_now.spk_id.first;
                                    int neu_id_now = E_now.neu_id.first;
                                    Event_unit event(T_now, E_now.spk_id, E_now.neu_id, false == W_fb[neu_id_now][i]);
                                    #pragma omp critical
                                    {
                                        Event_queue.push(event);
                                    }
                                }
    #endif
                                }
    #if defined(TRAIN_DFA)
                                for (const auto& E_now : Event_vec_now) {
                                    int spk_id_now = E_now.spk_id.first;
                                    int neu_id_now = E_now.neu_id.first;
                                    Event_unit event(T_now, E_now.spk_id, E_now.neu_id, false == W_fb[neu_id_now][i]);
                                    #pragma omp critical
                                    {
                                        Event_queue.push(event);
                                    }
                                }
    #endif
                            }
                        }
                        Neu_out[i].reset();
                        #pragma omp atomic
                        Neu_acc[static_cast<size_t>(i / N_out_times)] += 1;
                    }
#if defined(TRAIN_PHASE)
                // not firing but give the chance to negative update
                // there is missing case on on-train-phase
                // if SG is on, then the SG_ref would be flagged
                // only ref 4 case ! and training_singal is 4 !
                else if (enabling_train && !train_signal && Neu_out[i].is_ref(T_now) && Neu_out[i].is_SG_ref(T_now) && (static_cast<size_t>(i / N_out_times) != class_now) && (static_cast<size_t>(i % N_out_times) == train_index) ){
                    for (const auto& S_now : S_vec_now) {
                        int id_now = S_now.id.first;
                        char layer = S_now.id.second;

                        if (layer == 'r') {
                            std::pair<int, char> spk_id = std::make_pair(id_now, 'r');
                            std::pair<int, char> neu_id = std::make_pair(i, 'o');
                            Event_unit event(T_now, spk_id, neu_id, false);
                            #pragma omp critical
                            {
                                Event_queue.push(event);
                            }
                        }
                    }
                }
#endif
            }
            }

            #pragma omp section
#if defined(TRAIN_PHASE)
            if (enabling_train && !train_signal) {
#else
            if (enabling_train) {
#endif
                #pragma omp parallel for
                for (int k = 0; k < N_out_times; ++k) {
        #if defined(REFRACTORY)
                    if (!Neu_out[class_now * N_out_times + k].is_firing() && !Neu_out[class_now * N_out_times + k].is_ref(T_now) && k == train_index && !Neu_out[class_now * N_out_times + k].is_SG_ref(T_now)) {
        #endif
                        bool SG_now = Neu_out[class_now * N_out_times + k].get_SG();
                        if (SG_now) {
                            for (const auto& S_now : S_vec_now) {
                                int id_now = S_now.id.first;
                                char layer = S_now.id.second;

                                if (layer == 'r') {
                                    std::pair<int, char> spk_id = std::make_pair(id_now, 'r');
                                    std::pair<int, char> neu_id = std::make_pair(class_now * N_out_times + k, 'o');
                                    Event_unit event(T_now, spk_id, neu_id, true);
                                    #pragma omp critical
                                    {
                                        Event_queue.push(event);
                                    }
                                }
                            }
#if defined(TRAIN_ELIGIBLETRACE)
                            for (const auto& S_now : S_vec_trace_now) {
                                int id_now = S_now.id.first;
                                char layer = S_now.id.second;

                                if (layer == 'r') {
                                    std::pair<int, char> spk_id = std::make_pair(id_now, 'r');
                                    std::pair<int, char> neu_id = std::make_pair(class_now * N_out_times + k, 'o');
                                    Event_unit event(T_now, spk_id, neu_id, true);
                                    #pragma omp critical
                                    {
                                        Event_queue.push(event);
                                    }
                                }
                            }
#endif
#if defined(TRAIN_FA)
                            /*FA*/
                            for (const auto& E_now : Event_vec_now) {
                            int spk_id_now = E_now.spk_id.first;
                            int neu_id_now = E_now.neu_id.first;
                            Event_unit event(T_now, E_now.spk_id, E_now.neu_id, true == W_fb[neu_id_now][class_now*N_out_times + k]);
                            #pragma omp critical
                            {
                                Event_queue.push(event);
                            }
                        }
#endif
                        }
#if defined(TRAIN_DFA)
                        /*DFA*/
                        for (const auto& E_now : Event_vec_now) {
                            int spk_id_now = E_now.spk_id.first;
                            int neu_id_now = E_now.neu_id.first;
                            Event_unit event(T_now, E_now.spk_id, E_now.neu_id, true == W_fb[neu_id_now][class_now*N_out_times + k]);
                            #pragma omp critical
                            {
                                Event_queue.push(event);
                            }
                        }
#endif
                    }
                }
            }
        }

        if (enabling_train && !train_signal) {

            // std::cout << "Here is training part"<< std::endl;
            std::vector<Event_unit> events;
            #pragma omp critical
            {
                while (!Event_queue.empty()) {
                    events.push_back(Event_queue.top());
                    Event_queue.pop();
                }
            }

            // std::cout << "events.size(): "<< events.size() << std::endl;

            #pragma omp parallel for
            for (std::size_t i = 0; i < events.size(); ++i) {
                Event_unit& E_now = events[i];
                int spk_id_now = E_now.spk_id.first;
                char spk_l_now = E_now.spk_id.second;
                int neu_id_now = E_now.neu_id.first;
                char neu_l_now = E_now.neu_id.second;
                bool sign = E_now.sign;
                // std::cout << "Before update: W_out[" << spk_id_now << "][" << neu_id_now << "] = " << W_out[spk_id_now][neu_id_now] << std::endl;
                if (spk_l_now == 'r' && neu_l_now == 'o') {
                    if (sign) {
                        #pragma omp atomic
                        W_out[spk_id_now][neu_id_now] += lr*0.1;
                        if (W_out[spk_id_now][neu_id_now] > 1.0*0.1) W_out[spk_id_now][neu_id_now] = 1.0*0.1;
                    } else {
                        #pragma omp atomic
                        W_out[spk_id_now][neu_id_now] -= lr*0.1;
                        if (W_out[spk_id_now][neu_id_now] < -1.0*0.1) W_out[spk_id_now][neu_id_now] = -1.0*0.1;
                    }
                }
                // std::cout << "After update: W_out[" << spk_id_now << "][" << neu_id_now << "] = " << W_out[spk_id_now][neu_id_now] << std::endl;
#if defined(TRAIN_FA) || defined(TRAIN_DFA)

                else if (spk_l_now == 'r' && neu_l_now == 'r'){
                    if (sign) {
                        #pragma omp atomic
                        W_res[spk_id_now][neu_id_now] += lr*0.1;
                        if (W_res[spk_id_now][neu_id_now] > 0.10) W_res[spk_id_now][neu_id_now] = 0.1;
                    } else {
                        #pragma omp atomic
                        W_res[spk_id_now][neu_id_now] -= lr*0.1;
                        if (W_res[spk_id_now][neu_id_now] < -0.10) W_res[spk_id_now][neu_id_now] = -0.1;
                    }
                }
                /*
                else if (spk_l_now == 'i' && neu_l_now == 'r'){
                    if (sign) {
                        #pragma omp atomic
                        W_in[spk_id_now][neu_id_now] += lr*0.1;
                        if (W_in[spk_id_now][neu_id_now] > 0.1) W_in[spk_id_now][neu_id_now] = 0.1;
                    } else {
                        #pragma omp atomic
                        W_in[spk_id_now][neu_id_now] -= lr*0.1;
                        if (W_in[spk_id_now][neu_id_now] < -0.1) W_in[spk_id_now][neu_id_now] = -0.1;
                    }
                }

                else if (spk_l_now == 'r' && neu_l_now == 'r'){
                    if (sign) {
                        #pragma omp atomic
                        W_res[spk_id_now][neu_id_now] += lr*0.1;
                        if (W_res[spk_id_now][neu_id_now] > 1.0*0.1) W_res[spk_id_now][neu_id_now] = 1.0*0.1;
                    } else {
                        #pragma omp atomic
                        W_res[spk_id_now][neu_id_now] -= lr*0.1;
                        if (W_res[spk_id_now][neu_id_now] < -1.0*0.1) W_res[spk_id_now][neu_id_now] = -1.0*0.1;
                    }
                }*/
#endif
            }
            events.clear();
        }

        /*
        // L2 regularization
        if (enabling_train && !train_signal) {
            #pragma omp parallel for
            for (size_t j = 0; j < Neu_res.size(); ++j) {
                for (size_t i = 0; i < Neu_out.size(); ++i) {
                    #pragma omp atomic
                    W_out[i][j] -= lr * 0.1 * W_out[i][j];

                    //if (W_out[i][j] > 0) W_out[i][j] -= lr
                    //else W_out[i][j] += lr

                }
            }
        }*/

        S_vec_now.clear();
        S_vec_trace_now.clear();
        S_vec_trace_delay_now.clear();
        Event_vec_now.clear();

        S_vec_now.shrink_to_fit();
        S_vec_trace_now.shrink_to_fit();
        S_vec_trace_delay_now.shrink_to_fit();
        Event_vec_now.shrink_to_fit();
    }
#if defined(TRAIN_PHASE)
    PTE_slide = static_cast<size_t>((PTE_slide + 1) % PTE_times);
#endif
    // train_index = (train_index + 1) % N_out_times;

    uint8_t max_index = std::distance(Neu_acc.begin(), std::max_element(Neu_acc.begin(), Neu_acc.end()));
    bool is_correct = (max_index == class_now);
    return is_correct;
}

// Load spike train
void Core::load_spike_train(const std::vector<uint32_t>& spike_times, const std::vector<uint16_t>& neuron_indices) {
    while (!external_S_queue.empty()) external_S_queue.pop();
    for (size_t i = 0; i < spike_times.size(); ++i) {
        if (neuron_indices[i] < 144) external_S_queue.push(Spike(spike_times[i], {neuron_indices[i], 'i'}));
        else external_S_queue.push(Spike(spike_times[i], {neuron_indices[i] - 144, 'b'}));
    }
}

// Record spike
void Core::record_spike(uint32_t time, int neuron_index) {
    recorded_times.push_back(time);
    recorded_neuron_indices.push_back(static_cast<uint16_t>(neuron_index));
}

// Save weights to a file
void Core::save_weights(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for saving weights");
    }

    json weights_json;
    weights_json["W_in"] = W_in;
    weights_json["W_res"] = W_res;
    weights_json["W_out"] = W_out;
    weights_json["W_bias"] = W_bias;

    file << weights_json.dump(4);
    file.close();
}

// Load weights from a file
void Core::load_weights(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for loading weights");
    }

    json weights_json;
    file >> weights_json;

    W_in = weights_json["W_in"].get<std::vector<std::vector<double>>>();
    W_res = weights_json["W_res"].get<std::vector<std::vector<double>>>();
    W_out = weights_json["W_out"].get<std::vector<std::vector<double>>>();

    file.close();
}
