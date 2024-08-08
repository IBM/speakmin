#ifndef NEURON_H
#define NEURON_H

#include <cmath>

class Neuron {
private:
    double V_mem;      // Membrane potential
    double tau;        // Time constant
    double V_th;       // Threshold potential
    double V_reset;    // Reset potential
    uint32_t T_now;      // Current simulation time
    uint32_t T_last;     // Time of the last update for leaky computation
    double SG_window;  // Window for computing surrogate gradient
    uint32_t T_SG;       // Time for surrogate gradient calculation

#if defined(REFRACTORY)
    uint32_t T_ref;      // End of refractory period
    uint32_t t_ref;      // Duration of refractory period
#endif

public:
#if defined(REFRACTORY)
    Neuron(double V_init, double tau, double V_th, double V_reset, uint32_t t_ref, double SG_window)
        : V_mem(V_init), tau(tau), V_th(V_th), V_reset(V_reset), T_now(0), T_last(0), SG_window(SG_window), T_SG(0), T_ref(0), t_ref(t_ref) {}
#else
    Neuron(double V_init, double tau, double V_th, double V_reset, double SG_window)
        : V_mem(V_init), tau(tau), V_th(V_th), V_reset(V_reset), T_now(0), T_last(0), SG_window(SG_window), T_SG(0) {}
#endif

    // Getter methods
    double get_tau() const { return tau; }
    double get_V_th() const { return V_th; }
    double get_V_reset() const { return V_reset; }
    double get_V_mem() const { return V_mem; }
    double get_SG_window() const { return SG_window; }
    uint32_t get_T_last() const { return T_last; }

#if defined(REFRACTORY)
    uint32_t get_t_ref() const { return t_ref; }
#endif

    // Leaky function to update the membrane potential with exponential decay
    inline void leak(uint32_t current_time) {
        T_now = current_time;
        if (T_now == T_last) return;
        V_mem *= std::exp(-(static_cast<double>(T_now - T_last)) / tau);
        T_last = T_now;
    }

    // Function to apply an external input
    inline void in(double input) {
#if defined(REFRACTORY)
        if (T_now < T_ref) {
            return;
        }
#endif
        V_mem += input;
    }

    // Function to check if the neuron is firing
    inline bool is_firing() const { return V_mem >= V_th; }

    // Function to reset the membrane potential to the resting state after firing
    inline void reset() {
        V_mem = V_reset;
#if defined(REFRACTORY)
        T_ref = T_now + t_ref;
#endif
    }

    // Function to compute surrogate gradient
    inline bool get_SG() {
        bool SG = std::abs(V_mem - V_th) < SG_window;
        if (SG) T_SG = T_now + tau;  // Assuming you meant tau, not t_ref
        return SG;
    }

    // Function to check if surrogate gradient reference time is active
    inline bool is_SG_ref(uint32_t T_now) const { return T_now < T_SG; }

#if defined(REFRACTORY)
    inline bool is_ref(uint32_t T_now) const { return T_now < T_ref; }
#endif
};

#endif // NEURON_H
