# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

SMSIM ?= ../src/SMsim

# fixed params
LOG_FILE = accuracy_log.csv
JSON_FILES += init_parameters.json
JSON_FILES += init_weights.json
JSON_FILES += init_taus.json

run : ${LOG_FILE}

${LOG_FILE} : ${JSON_FILES} ${SMSIM}
	${SMSIM}

${JSON_FILES} :
	python3 gen_config.py

${SMSIM} :
	make -C ../src

clean :
	-rm ${JSON_FILES}
	-rm ${LOG_FILE}

.PHONY: run
