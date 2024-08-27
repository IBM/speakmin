CATEGORY ?= up
FILE_INDEX ?= 1
# FILE_INDEX must starts from 1 (not 0)
N_MELS ?= 16
ALPHA ?= 1.0
VTH ?= 0.01
ECHO ?= 1
ALIGN ?= 1
TIME_UNIT ?= 1.0

GIT_HOME ?= $(realpath ../..)
DATASET_PATH ?= ${GIT_HOME}/dataset/speech_commands_v0.02

PYTHONPATH := ${GIT_HOME}/python:${PYTHONPATH}
export PYTHONPATH

AUDIO_FILE = $(word ${FILE_INDEX},$(wildcard ${DATASET_PATH}/${CATEGORY}/*.wav))
OUTPUT_BASE = ${CATEGORY}_$(shell printf '%05g' $(FILE_INDEX))_$(basename $(notdir ${AUDIO_FILE}))
OUTPUT_PATH = ${OUTPUT_BASE}
PICKLE_FILE = ${OUTPUT_PATH}/${OUTPUT_BASE}.pickle
PLOT_FILE = ${OUTPUT_PATH}/data_original.png

PY_SCRIPT_PLOTS = gen_plots.py
PY_SCRIPT_SPIKES = gen_spikes.py

ifeq ($(ECHO),1)
  PY_OPTS += --echo
endif

ifeq ($(ALIGN),1)
  PY_OPTS += --align
endif

all : spikes plots

spikes : ${PICKLE_FILE}

plots : ${PLOT_FILE}

${PICKLE_FILE} : ${AUDIO_FILE} ${OUTPUT_PATH}
	python3 ${PY_SCRIPT_SPIKES} $< $@ --n_mels ${N_MELS} --alpha ${ALPHA} --vth ${VTH} --time_unit ${TIME_UNIT} ${PY_OPTS}

${PLOT_FILE} : ${AUDIO_FILE} ${OUTPUT_PATH}
	python3 ${PY_SCRIPT_PLOTS} $< --n_mels ${N_MELS} --alpha ${ALPHA} --vth ${VTH} --output_path ${OUTPUT_PATH} --time_unit ${TIME_UNIT} ${PY_OPTS}

${OUTPUT_PATH} :
	mkdir -p $@

clean :
	@rm -rf ${OUTPUT_PATH}

.PHONY : clean all spikes plots
