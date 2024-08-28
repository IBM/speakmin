// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#ifndef SPIKE_H
#define SPIKE_H

#include <cstddef>  // for size_t
#include <cstdint>
#include <utility>

class Spike {
public:
    Spike(uint32_t time, std::pair<size_t, char> id);

    bool operator<(const Spike& other) const;

    uint32_t time;
    std::pair<size_t, char> id;
};

class Spike_input {
public:
    Spike_input(uint32_t time, uint16_t id);

    bool operator<(const Spike_input& other) const;

    uint32_t time;
    uint16_t id;
};

#endif // SPIKE_H
