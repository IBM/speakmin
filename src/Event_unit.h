#include <utility> // For std::pair
#include <iostream>

class Event_unit {
    // for computing spatial gradient 
public:
    double time;                  // Spiking time
    std::pair<size_t, char> spk_id;  // Neuron ID (number and side)
    std::pair<size_t, char> neu_id;  // Neuron ID (number and side)
    bool sign;                    // delta, sign from Surrogate Gradient 
    // float delta;               // if we use accumulative neuron

    // Constructor
    // Event_unit(double time, std::pair<size_t, char> spk_id,  std::pair<size_t, char> neu_id, bool SG);
    Event_unit(double time, std::pair<size_t, char> spk_id,  std::pair<size_t, char> neu_id, bool sign);

    // Overload < operator to use in priority queue
    bool operator<(const Event_unit& other) const {
        return time < other.time; // For sorting in descending order, use >
    }
};

/*
class Event_unit_all {
    // for computing 3 components of gradients, spatial / temporal / spatial-temporal gradient 
public:
    double time;                  // Spiking time
    std::pair<size_t, char> spk_id;  // Neuron ID (number and side)
    std::pair<size_t, char> neu_id;  // Neuron ID (number and side)
    bool sign;                    // delta, sign from Surrogate Gradient 
    // float delta;               // if we use accumulative neuron
    double V_mem_prev;            // U[t-1]
    double V_leak;                // leak[t-tau]
    double V_leak_prev;           // leak[t-tau+1]

    // Constructor
    Event_unit_all(double time, std::pair<size_t, char> spk_id, std::pair<size_t, char> neu_id, bool sign, double V_mem_prev, double V_leak, double V_leak_prev);
    
    // Overload < operator to use in priority queue
    bool operator<(const Event_unit_all& other) const {
        return time < other.time; // For sorting in descending order, use >
    }
};
*/
