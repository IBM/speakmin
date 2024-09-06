## How to run
1. Download dataset (one time)
```
$ cd ${S2S_HOME}/dataset
$ make
```
2. Generate PWL file
```
$ make {CATEGORY=} {FILE_INDEX=} {VSCALE=} {VSHIFT=} {ECHO=} {DATASET_PATH=}
```

|PARAM   |DEFAULT|OPTIONS| DESCRIPTION     |
|:-------|:----------------|:--------|:--------|
|CATEGORY|up|backwawrd,<br>bed,<br>bird,<br>cat,<br>dog,<br>down,<br>eight,<br>five,<br>follow,<br>forward,<br>four,<br>go,<br>happy,<br>house,<br>learn,<br>left,<br>marvin,<br>nine,<br>no,<br>off,<br>on,<br>one,<br>right,<br>seven,<br>sheila,<br>six,<br>stop,<br>three,<br>tree,<br>two,<br>up,<br>visual,<br>wow,<br>yes,<br>zero|Dataset category |
|FILE_INDEX|1|any |File index number in each category. Must start from `1` not `0`|
|VSCALE |0.0894e-3|any|Voltage scaling from original data |
|VSHIFT |0.0|any|Voltage shift from original data |
|ECHO   |1|0,1|Echo information |
|DATASET_PATH|${S2S_HOME}/dataset/speech_commands_v0.02|any |Dataset path |
