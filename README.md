# speakmin_project
This repository is for open-sourcing process.
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

1. Make speak2spike(s2s) dataset

```
$ cd /speakmin_project/speech2spikes/tools
$ pip install -r requirements.txt
$ cd /speakmin_project/speech2spikes/tools/gen_spike
$ make OUTPUT_FILES="train0.bin train1.bin train2.bin train3.bin train4.bin train5.bin train6.bin train7.bin train8.bin train9.bin" WAV_FILE_SOURCE=not_testing SPLIT_NUM="300 300 300 300 300 300 300 300 300 300 " CATEGORY="yes no up down left right on off stop go"
$ make OUTPUT_FILES="test.bin " WAV_FILE_SOURCE=testing SPLIT_NUM="300 " CATEGORY="yes no up down left right on off stop go"
```


1. Run `gen_config.py` to generate the configuration files in the Makefile directory.

```
$ pip install -r requirements.txt
$ cd /speakmin_project
$ python gen_config.py
```

2. Run SM simulation

```
$ cd /speakmin_project/src
$ make run
```
