## How to run
1. Download dataset (one time)
```
$ cd ${GIT_HOME}/dataset
$ make
```
2. Generate PWL file
```
$ make {CATEGORY=} {FILE_INDEX=} {VSCALE=} {VSHIFT=} {ECHO=} {DATASET_PATH=}
```

|PARAM   | DESCRIPTION     | OPTIONS | DEFAULT |
|:-------|:----------------|:--------|:--------|
|CATEGORY|Dataset category |backwawrd,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero|up|
|FILE_INDEX|File index number in each category. Must start from `1` not `0`|any |1|
|VSCALE |Voltage scaling from original data |any|0.0894e-3|
|VSHIFT |Voltage shift from original data |any|0.0|
|ECHO   |Echo information |0,1|1|
|DATASET_PATH|Dataset path |any |${GIT_HOME}/dataset/speech_commands_v0.02|
