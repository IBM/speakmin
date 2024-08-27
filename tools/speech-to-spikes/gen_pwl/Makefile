CATEGORY ?= up
FILE_INDEX ?= 1
# FILE_INDEX must starts from 1 (not 0)
VSCALE ?= 0.0894e-3
VSHIFT ?= 0.0
ECHO ?= 1

S2S_HOME ?= $(realpath ..)
DATASET_PATH ?= ${S2S_HOME}/dataset/speech_commands_v0.02

PYTHONPATH := ${S2S_HOME}/python:${PYTHONPATH}
export PYTHONPATH

AUDIO_FILE = $(word ${FILE_INDEX},$(sort $(wildcard ${DATASET_PATH}/${CATEGORY}/*.wav)))
PWL_FILE = ${CATEGORY}_$(shell printf '%05g' $(FILE_INDEX))_$(basename $(notdir ${AUDIO_FILE})).inc

PY_SCRIPT = gen_pwl.py

ifeq ($(ECHO),1)
  PY_OPTS += --echo
endif

${PWL_FILE} : ${AUDIO_FILE}
	python3 ${PY_SCRIPT} $< $@ --vscale ${VSCALE} --vshift ${VSHIFT} ${PY_OPTS}

clean :
	@rm ${PWL_FILE}

.PHONY : clean
