# speakmin
`speakmin` is an event-based simulator for spiking reservoir computing. It includes core functionalities for simulating neurons, handling spikes, and recording events. Keyword spotting is implemented as one of demonstrations, but any demonstrations can be run if spike dataset is prepared. The project is built using C++17 and leverages OpenMP for parallel processing.

> ⚠️ This repository is currently in beta and under active development. Please be mindful of potential issues and keep an eye for improvements, new features and bug fixes in upcoming versions.

## Prerequisites

- **C++17** compatible compiler (e.g., `g++`)
- **OpenMP** support for parallelization
- **Python** 3.10.12 or higher
- **nlohmann/json** read the .json files
```bash
### Download json.hpp ###
$ mkdir -p include/nlohmann
$ wget https://github.com/nlohmann/json/releases/latest/download/json.hpp -P include/nlohmann/
  or
$ curl -L -o include/nlohmann/json.hpp https://github.com/nlohmann/json/releases/latest/download/json.hpp

### Install python modules ###
$ pip install -r requirements.txt
```

## File trees
```bash
python               # Directory for Python scripts
src                  # Directory for source code
├─ SMsim.cpp         # Main simulation file
├─ Core.cpp          # Core functionalities of the simulator
├─ Event_unit.cpp    # Event handling functionalities
├─ Spike.cpp         # Spike handling functionalities
└─ Makefile          # Makefile to build and manage the project
tools                # Directory for tools
└─ speech-to-spikes  # Directory for speech-to-spike converstion utility
run                  # Directory for running simulations (base)
├─ gen_config.py     # Script to generate sim configuration
└─ Makefile          # Makefile for running simulations
```

## How to run
### 1. Generates spike dataset

See details in `tools/speech-to-spikes/gen_spike/readme.md`. Followings are examples.

```bash
### Training dataset ###
$ cd tools/speech-to-spikes/gen_spike
$ make OUTPUT_FILES="train0.bin train1.bin train2.bin train3.bin train4.bin train5.bin train6.bin train7.bin train8.bin train9.bin" WAV_FILE_SOURCE=not_testing SPLIT_NUM="300 300 300 300 300 300 300 300 300 300 " CATEGORY="yes no up down left right on off stop go" ALPHA=10 LEAK_ENABLE=1 LEAK_TAU=20000e-6

### Test dataset ###
$ cd tools/speech-to-spikes/gen_spike
$ make OUTPUT_FILES="test.bin " WAV_FILE_SOURCE=testing SPLIT_NUM="300 " CATEGORY="yes no up down left right on off stop go" ALPHA=10 LEAK_ENABLE=1 LEAK_TAU=20000e-6
```

### 2. Compiles codes
```bash
$ cd src
$ make TRAIN_MODE={DFA|FA|NONE} TRAIN_PHASIC_ENABLED={0|1} TRAIN_ELIGIBLETRACE_ENABLED={0|1}
```
- Compile options

|Parameter |Default|Options |Description      |
|:---------|:------|:-------|:----------------|
|TRAIN_MODE|DFA    |DFA, FA, NONE|Training mode. DFA: Direct Feedback Alignment (`TRAIN_DFA`), FA: Feedback Alignment (`TRAIN_FA`) |
|TRAIN_PHASIC_ENABLED|0     |0,1       |If `1`, phasic operations are enabled (`TRAIN_PHASE` will be defined)|
|TRAIN_ELIGIBLETRACE_ENABLED|0 |0,1 |If `1`, eligibile trace function is enabled (`TRAIN_ELIGIBLETRACE`)|

- Additional Makefile Targets 

```bash
### Removes the compiled object files and the executable.
$ make clean

### Compiles the project with debugging symbols enabled.
$ make debug   

### Installs the executable to `INSTALL_DIR` (default:/usr/local/bin).
$ make install {INSTALL_DIR=/desired/path/to/install}

### Removes the installed executable from /usr/local/bin.
$ make uninstall
```

### 3. Runs simulations
```bash
$ cp -rp run run_1 (run_1 can be any your preferred name)
$ cd run_1
(modify gen_config.py if necessary)
$ make
```

## License
This project is licensed under Apache License 2.0.
