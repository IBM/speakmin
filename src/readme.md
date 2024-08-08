# SpeakMin Project

This project implements a spiking neural network simulator called SpeakMin. It includes core functionalities for simulating neurons, handling spikes, and recording events. The project is built using C++17 and leverages OpenMP for parallel processing.

## Prerequisites

- GCC compiler with C++17 support
- OpenMP
- Git

## Project Structure

| File            | Description                              |
|-----------------|------------------------------------------|
| `SMsim.cpp`     | Main simulation file                     |
| `Core.cpp`      | Core functionalities of the simulator    |
| `Event_unit.cpp`| Event handling functionalities           |
| `Spike.cpp`     | Spike handling functionalities           |
| `include/`      | Header files directory                   |
| `Makefile`      | Makefile to build and manage the project |

## Building the Project

1. Run `config_gen_s2s_bias_rsnn.py` to generate the configuration files in the Makefile directory.
2. Navigate to the Makefile directory and run:
