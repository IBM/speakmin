# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

OUTPUT_FILES ?= train.bin test.bin
PICKLE_FILE ?= $(addsuffix .pickle,$(basename $(word 1,$(OUTPUT_FILES))))

WINDOW_SIZE ?= 100000
WINDOW_STRIDE ?= 50000
PLOT_NUMBER ?= 16
PLOT_INDEX_FROM ?= 0
PLOT_INDEX_LIST ?= None

S2S_HOME ?= $(realpath ..)

PYTHONPATH := ${S2S_HOME}/python:${PYTHONPATH}
export PYTHONPATH

PY_SCRIPT_SPIKES = analysis.py

ifneq ($(PLOT_INDEX_LIST),None)
  PY_OPTS += --plot_index_list ${PLOT_INDEX_LIST}
endif

ARCHIVE_PATH ?= backup
ARCHIVE_TARGET_FILES += $(wildcard *_compressed_data.png)
ARCHIVE_TARGET_FILES += c2c_distance.png
ARCHIVE_TARGET_FILES += c2c_distance_sd.png
ARCHIVE_TARGET_FILES += class_sd.md
ARCHIVE_FILES = $(addprefix ${ARCHIVE_PATH}/,${ARCHIVE_TARGET_FILES})

analysis :
	python3 ${PY_SCRIPT_SPIKES} ${PICKLE_FILE} --window_size ${WINDOW_SIZE} --window_stride ${WINDOW_STRIDE} \
	--plot_number ${PLOT_NUMBER} --plot_index_from ${PLOT_INDEX_FROM} ${PY_OPTS}

archive : ${ARCHIVE_FILES}

${ARCHIVE_PATH} :
	mkdir -p $@

${ARCHIVE_FILES} : ${ARCHIVE_PATH}
	-@mv $(subst ${ARCHIVE_PATH}/, ,$@) $@

clean :
	-@rm ${ARCHIVE_TARGET_FILES}

.PHONY : clean plots archive
