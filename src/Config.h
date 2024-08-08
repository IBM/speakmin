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
    double tau;
    double V_th;
    double V_reset;
#if defined(REFRACTORY)
    double t_ref;
#endif
    double SG_window;
    
    int N_in;
    int N_res;
    int N_out;
    int N_bias;
};

#endif // CONFIG_H
