#include "Spike.h"

// Constructor
// example of Neu_id: # of neuron, site (0, 'i') or (0, 'r'). 
// 'i' is 'input' and 'r' is 'reservoir'
// or 'f' is 'forward' (A side of CBA) and 'b' is 'hidden' (b side of CBA) 
Spike::Spike(uint32_t time, std::pair<size_t, char> id) : time(time), id(id) {}

// Comparison operator for priority queue
bool Spike::operator<(const Spike& other) const {
    return time > other.time;  // Higher priority for earlier times
}

Spike_input::Spike_input(uint32_t time, uint16_t id) : time(time), id(id){}

// Comparison operator for priority queue
bool Spike_input::operator<(const Spike_input& other) const {
    return time > other.time;  // Higher priority for earlier times
}
