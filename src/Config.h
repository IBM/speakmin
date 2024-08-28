// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#ifndef CONFIG_H
#define CONFIG_H

#include <vector>

struct Config {
    std::vector<std::vector<double>> W_in;
    std::vector<std::vector<double>> W_res;
    std::vector<std::vector<double>> W_out;
    std::vector<std::vector<bool>> W_fb;
    std::vector<std::vector<double>> W_bias;
    double T_sim;
    double t_delay;
    double V_init;
    double tau_out;
    double V_th;
    double V_bot;
    double V_reset;
#if defined(REFRACTORY)
    double t_ref;
#endif
    double SG_window;

    int N_in;
    int N_res;
    int N_out;
    int N_class;
    int N_out_times;
    int N_bias;

    size_t ET_N;
    size_t PTE_slide;
    size_t PTE_times;
    size_t PTE_range;
};

#endif // CONFIG_H
