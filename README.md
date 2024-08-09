# speakmin
This project implements a spiking reservoir computing simulator for keyword spotting named SpeakMin. 
It includes core functionalities for simulating neurons, handling spikes, and recording events. The project is built using C++17 and leverages OpenMP for parallel processing.

## Prerequisites

- **C++17** compatible compiler (e.g., `g++`)
- **OpenMP** support for parallelization
- **nlohmann/json** read the .json files
```
wget https://github.com/nlohmann/json/releases/latest/download/json.hpp -P include/nlohmann/
```
## Project Structure

| File            | Description                              |
|-----------------|------------------------------------------|
| `include/`      | Header files directory                   |
| `speech2spikes/`      | speech2spike dataset                   |
| `src`           |                                          |
| `├── SMsim.cpp`     | Main simulation file                     |
| `├── Core.cpp`      | Core functionalities of the simulator    |
| `├── Event_unit.cpp`| Event handling functionalities           |
| `├── Spike.cpp`     | Spike handling functionalities           |
| `└── Makefile`      | Makefile to build and manage the project |
| `gen_config.py` | Script to generate configuration         |

## Generating the Dataset

1. Make speak2spike (s2s) dataset

```
cd /speakmin_project/speech2spikes/tools
```
```
pip install -r requirements.txt
```
```
cd /speakmin_project/speech2spikes/tools/gen_spike
```
generate s2s trianing dataset
```
make OUTPUT_FILES="train0.bin train1.bin train2.bin train3.bin train4.bin train5.bin train6.bin train7.bin train8.bin train9.bin" WAV_FILE_SOURCE=not_testing SPLIT_NUM="300 300 300 300 300 300 300 300 300 300 " CATEGORY="yes no up down left right on off stop go"
```
generate s2s test dataset
```
make OUTPUT_FILES="test.bin " WAV_FILE_SOURCE=testing SPLIT_NUM="300 " CATEGORY="yes no up down left right on off stop go"
```

## Building the Project

1. Run `gen_config.py` to generate the configuration files in the Makefile directory.

```
pip install -r requirements.txt
```
```
cd /speakmin_project
```
```
python gen_config.py
```

2. Run SM simulation

```
cd /speakmin_project/src
```
```
make run
```
you can install on the preferred path where you want. The default path is `/usr/local/bin`.
```
make install INSTALL_DIR=/desired/path/to/install
```

## Training Modes

- **DFA (Direct Feedback Alignment):** The 'default' training mode if no other mode is specified.
```
make run TRAIN_MODE=DFA
```
- **FA (Feedback Alignment):** Can be activated by setting the `TRAIN_MODE` variable to `FA`.
```
make run TRAIN_MODE=FA
```

### Optional Features

- **TRAIN_PHASE:** This feature can be enabled by setting the `TRAIN_PHASIC_ENABLED` flag.
```
make run TRAIN_PHASIC_ENABLED=1
```
- **TRAIN_ELIGIBLETRACE:** This feature can be enabled by setting the `TRAIN_ELIGIBLETRACE_ENABLED` flag.
```
make run TRAIN_ELIGIBLETRACE_ENABLED=1
```

### Additional Makefile Targets `make clean`, `make debug`, `make install`, `make uninstall`

Removes the compiled object files and the executable.
```
make clean
```

Compiles the project with debugging symbols enabled.
```
make debug
```

Installs the executable to /usr/local/bin.
```
make install
```
Removes the installed executable from /usr/local/bin.
```
make uninstall
```


## License
This project is licensed under Apache License 2.0.