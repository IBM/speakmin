OUTPUT_FILES ?= train.bin test.bin
CATEGORY ?= zero one two
SPLIT_NUM ?= 16 16
VTH ?= 0.01
ALIGN ?= 1
NORM ?= 1
WAV_FILE_SOURCE ?= all
USE_ALL_IN_SOURCE ?= 0
LEAK_ENABLE ?= 0
LEAK_TAU ?= 16000e-6
PREEMPHASIS ?= 0
PREEMPHASIS_COEF ?= 0.97

N_MELS ?= 16
MEL_NORM ?= slaney
ALPHA ?= 1.0
TIME_UNIT ?= 1e-6
RNG_SEED ?= 10
ECHO ?= 0
NORM_TARGET_DBFS ?= -31.782
NUM_PROCESS ?= 8
USE_VCSV_FILE ?= 0
VCSV_FILE_PREFIX ?= c
VCSV_FILE_SUFFIX ?= .vcsv
VCSV_FILE_LIST ?= $(addsuffix ${VCSV_FILE_SUFFIX},$(addprefix ${VCSV_FILE_PREFIX},$(shell seq -w 16)))
# c00.vcsv c01.vcsv ... c16.vcsv

GIT_HOME ?= $(realpath ../..)
DATASET_PATH ?= ${GIT_HOME}/dataset/speech_commands_v0.02

PYTHONPATH := ${GIT_HOME}/python:${PYTHONPATH}
export PYTHONPATH

PY_SCRIPT_SPIKES = speech2spikes.py

ifeq ($(USE_ALL_IN_SOURCE),1)
  PY_OPTS += --use_all_in_source
endif

ifeq ($(ALIGN),1)
  PY_OPTS += --align
endif

ifeq ($(ECHO),1)
  PY_OPTS += --echo
endif

ifeq ($(NORM),1)
  PY_OPTS += --norm
endif

ifeq ($(USE_VCSV_FILE),1)
  PY_OPTS += --vcsv_file_list ${VCSV_FILE_LIST}
endif

ifeq ($(LEAK_ENABLE),1)
  PY_OPTS += --leak_enable --leak_tau ${LEAK_TAU}
endif

ifeq ($(PREEMPHASIS),1)
  PY_OPTS += --preemphasis --preemphasis_coef ${PREEMPHASIS_COEF}
endif

ARCHIVE_PATH ?= backup
ARCHIVE_TARGET_FILES += ${OUTPUT_FILES}
ARCHIVE_TARGET_FILES += $(addsuffix .pickle,$(basename ${OUTPUT_FILES}))
ARCHIVE_TARGET_FILES += mel_freq_bank.png
ARCHIVE_TARGET_FILES += mel_freq_bank_log.png
ARCHIVE_FILES = $(addprefix ${ARCHIVE_PATH}/,${ARCHIVE_TARGET_FILES})

all : ${OUTPUT_FILES}

${OUTPUT_FILES} :
	python3 ${PY_SCRIPT_SPIKES} ${OUTPUT_FILES} --dataset_path ${DATASET_PATH} --category ${CATEGORY} \
	--split_number ${SPLIT_NUM} --n_mels ${N_MELS} --alpha ${ALPHA} --vth ${VTH} --time_unit ${TIME_UNIT} --rng_seed ${RNG_SEED} ${PY_OPTS} \
	--norm_target_dBFS ${NORM_TARGET_DBFS} --wav_file_source ${WAV_FILE_SOURCE} --mel_norm ${MEL_NORM} --num_process ${NUM_PROCESS}

archive : ${ARCHIVE_FILES}

${ARCHIVE_PATH} :
	mkdir -p $@

${ARCHIVE_FILES} : ${ARCHIVE_PATH}
	-@mv $(subst ${ARCHIVE_PATH}/, ,$@) $@

clean :
	-@rm ${ARCHIVE_TARGET_FILES}

.PHONY : clean all archive
