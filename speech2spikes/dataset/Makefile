# (C) Copyright IBM Corp. 2024
#
# Download Google Speech Command Dataset
#
VERSION ?= 2

DATASET_PATH_NAME = speech_commands_v0.0${VERSION}
DATASET_URL = http://download.tensorflow.org/data
TARGZ_FILE = ${DATASET_PATH_NAME}.tar.gz

${DATASET_PATH_NAME} : ${TARGZ_FILE}
	mkdir -p $@
	tar -xvf $< -C $@

${TARGZ_FILE} :
	curl ${DATASET_URL}/$@ -O

clean :
	rm ${TARGZ_FILE}
	rm -rf ${DATASET_PATH_NAME}

.PHONY : clean
