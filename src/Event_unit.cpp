// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#include "Event_unit.h"

// Constructor for Event_unit
Event_unit::Event_unit(double time, std::pair<size_t, char> spk_id, std::pair<size_t, char> neu_id, bool sign)
    : time(time), spk_id(spk_id), neu_id(neu_id), sign(sign) {}

/*
// Constructor for Event_unit_all
Event_unit_all::Event_unit_all(double time, std::pair<size_t, char> spk_id, std::pair<size_t, char> neu_id, bool sign, double V_mem_prev, double V_leak, double V_leak_prev)
    : time(time), spk_id(spk_id), neu_id(neu_id), sign(sign), V_mem_prev(V_mem_prev), V_leak(V_leak), V_leak_prev(V_leak_prev) {}
*/
