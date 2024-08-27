OUTPUT_FILES ?= train.bin test.bin
PICKLE_FILE ?= $(addsuffix .pickle,$(basename $(word 1,$(OUTPUT_FILES))))

PLOT_RASTER ?= 1
PLOT_WFORM ?= 1
PLOT_NUMBER ?= 16
PLOT_INDEX_FROM ?= 0
PLOT_INDEX_LIST ?= None
PLOT_TITLE ?= 1

S2S_HOME ?= $(realpath ..)

PYTHONPATH := ${S2S_HOME}/python:${PYTHONPATH}
export PYTHONPATH

PY_SCRIPT_SPIKES = plot_spikes.py

ifeq ($(PLOT_RASTER),1)
  PY_OPTS += --plot_raster
endif

ifeq ($(PLOT_WFORM),1)
  PY_OPTS += --plot_wform
endif

ifeq ($(PLOT_TITLE),1)
  PY_OPTS += --plot_title
endif

ifneq ($(PLOT_INDEX_LIST),None)
  PY_OPTS += --plot_index_list ${PLOT_INDEX_LIST}
endif

ARCHIVE_PATH ?= backup
ARCHIVE_TARGET_FILES += $(wildcard *.png)
ARCHIVE_FILES = $(addprefix ${ARCHIVE_PATH}/,${ARCHIVE_TARGET_FILES})

plots :
	python3 ${PY_SCRIPT_SPIKES} ${PICKLE_FILE} --plot_number ${PLOT_NUMBER} --plot_index_from ${PLOT_INDEX_FROM} ${PY_OPTS}

archive : ${ARCHIVE_FILES}

${ARCHIVE_PATH} :
	mkdir -p $@

${ARCHIVE_FILES} : ${ARCHIVE_PATH}
	-@mv $(subst ${ARCHIVE_PATH}/, ,$@) $@

clean :
	@rm *raster*.png
	@rm *wform*.png

.PHONY : clean plots archive
