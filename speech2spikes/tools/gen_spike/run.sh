#
# Test, all from "testing" source pool, categories are core 20 words.
#------------------------------------------------------------------------------
rm -rf test
make OUTPUT_FILES=test.bin WAV_FILE_SOURCE=testing USE_ALL_IN_SOURCE=1 \
     CATEGORY="zero one two three four five six seven eight nine yes no up down left right on off stop go"
make ARCHIVE_PATH=test archive

#
# Validation, all from "validation" source pool, categories are core 20 words.
#------------------------------------------------------------------------------
#rm -rf validation
#make OUTPUT_FILES=validation.bin WAV_FILE_SOURCE=validation USE_ALL_IN_SOURCE=1 \
#     CATEGORY="zero one two three four five six seven eight nine yes no up down left right on off stop go"
#make ARCHIVE_PATH=validation archive

#
# Training, all from "not_testing_validation" source pool, categories are core 20 words.
#------------------------------------------------------------------------------
rm -rf train
make OUTPUT_FILES=train.bin WAV_FILE_SOURCE=not_testing_validation USE_ALL_IN_SOURCE=1 \
     CATEGORY="zero one two three four five six seven eight nine yes no up down left right on off stop go"
make ARCHIVE_PATH=train archive

#
# Training and test, 128 for each from "all" source pool, categories are core 10/20 words.
#------------------------------------------------------------------------------
#rm -rf train128_test128_from_all
#make OUTPUT_FILES="train.bin test.bin" SPLIT_NUM="128 128" WAV_FILE_SOURCE=all \
#     CATEGORY="yes no up down left right on off stop go"
#make ARCHIVE_PATH=train128_test128_from_all archive

