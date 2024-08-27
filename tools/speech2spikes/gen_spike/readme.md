## How to run
```
$ make all [OPTION1=XXX OPTION2=YYY ...]
```
See OPTIONs in `Major option` and `Minor options` sections.

The target:`all` consists of sub targets `spikes`, `plots`, and `analysis`.
```
all
 +- spikes
 +- plots
 +- analysis
```

You can also run the sub targets independently such as below.
```
$ make -f speech2spikes.mk [OPTION=, OPTION=, ...]
$ make -f plot_spikes.mk [OPTION=, OPTION=, ...]
$ make -f analysis.mk [OPTION=, OPTION=, ...]
```

After you run `all`, you can archive the data such as below. The default archive path is `backup`. You can change it anything by using `ARCHIVE_PATH` param option.
```
$ make archive [ARCHIVE_PATH=]
```

If you want to run with different option(s) or rerun, do `clean` first.
```
$ make clean
```

See `run.sh` for some examples.

### Output file (.bin) data format

|Type|Num of bytes|Data|
|:---|:-----------|:---|
|Unsigned char |1 |Label index (see `Categories` table)|
|Unsigned short|2 |Data index in category              |
|Unsigned int  |4 |Unique global ID                    |
|Unsigned int  |4 |Number of data points (N)           |
|Unsigned char |1 |Reserved                            |
|Unsigned char |1 |Reserved                            |
|Unsigned int  |4 |Time for spike-0                    |
|Unsigned short|2 |Neuron index for spike-0            |
|Unsigned int  |4 |Time for spike-1                    |
|Unsigned short|2 |Neuron index for spike-1            |
|              |  | (continues)                        |
|Unsigned int  |4 |Time for spike-(N-1)                |
|Unsigned short|2 |Neuron index for spike-(N-1)        |

### Output file (.pickle) data format
List of dictionary data. Each dictionary data represents one input data which contains following keys.

|Key|Description|
|:--|:----------|
|label      |Index number of the category |
|uid        |Unique ID                    |
|category   |Name of category             |
|split_index|Index number for data-split (group index)|
|data_index |Index number of data in the split group of category|
|wav_file   |Original .wav file name                  |
|num_points |Number of spike points                   |
|spikes     |List of spikes. Element of the list represents one spike which contains of a list of [time, neuron_index]|

## Options
### Major options

When multiple values are defined as a list, use space as delimiter and enclose them in double quotation marks.

|Parameter   | Default      | Options    | Description  |
|:-----------|:-------------|:-----------|:-------------|
|OUTPUT_FILES|"train.bin test.bin"|any   |List of output file names.        |
|CATEGORY    |"zero one two"|(see "Categories" section)|List of categories. |
|SPLIT_NUM   |"16 16"       |any         |Split data into several groups. Default: "16 16" will devide data set into two groups which contain non-overlapped randomly selected 16 data for each.|
|VTH         |0.01          |any         |Threshold value for spike conversion. Lower value generates higher spike rate.|
|PREEMPHASIS |0             |0,1         |If `1`, waveform preemphasis will be done. `PREEMPHASIS_COEF` is used as a parameter.|
|ALIGN       |1             |0,1         |If `1`, waveform alignment will be done.|
|NORM        |1             |0,1         |If `1`, waveform amplitude will be normalized.|
|WAV_FILE_SOURCE|all        |all,testing,validation,not_testing,not_validation,not_testing_validation|which set of files is used for .wav file selection. `testing` uses .wav files listed in `testing_list.txt`. `validation` uses .wav files listed in `validation_list.txt`. `not_testing` uses .wav files not listed in `testing_list.txt`. `not_validation` uses .wav files not listed in `validation_list.txt`. `not_testing_validation` uses .wav files not listed in both `testing_list.txt` and `validation_list.txt`.|
|USE_ALL_IN_SOURCE|0        |0,1         |If `1`, all .wav files defined by `WAV_FILE_SOURCE` will be used. Numbers defined by `SPLIT_NUM` are ignored. Only one output spike file (.bin) is generated and the file name will be `OUTPUT_FILES[0]`.|
|NUM_PROCESS |8             |any         |Number of process for multithreading|
|USE_VCSV_FILE|0            |0,1         |If `1`, VCSV files extracted from circuit design will be used. `VCSV_FILE_LIST` must be defined if `USE_VCSV_FILE` is `1`.|
|VCSV_FILE_LIST|c01.vcsv c02.vcsv ... c16.vcsv|any        |List of VCSV file names to be used. This is valid only if `USE_VCSV_FILE` == `1`. Default file names will be `${VCSV_FILE_PREFIX}` + [01-16] + `${VCSV_FILE_SUFFIX}` (c00.vcsv c01.vcsv ... c16.vcsv). If `VCSV_FILE_LIST` is defined, the default name will be overwritten.|
|LEAK_ENABLE |0             |0,1         |Enable leak function in IF neuron.|
|LEAK_TAU    |16000e-6      |any         |Leak time constant in LIF neuron. This is valid only if `LEAK_ENABLE` == `1`.|


### Minor options
|Parameter   | Default | Options    | Description  |
|:-----------|:--------|:-----------|:-----------|
|N_MELS      |16       |any         |Number of channels for Mel-frequency banks.|
|MEL_NORM    |slaney   |slaney,None |Normalization method for Mel-freqyency banks. `None` creates flat-amplitude and peak amplitude will be `1`. To reduce the amplitude, use `ALPHA` as coefficient parameter. Around `ALPHA`=0.003 will create similar amplitude with `slaney`.|
|ALPHA       |1.0      |any         |Coefficient number when integrate filtered waveform for spike conversion.|
|RNG_SEED    |10       |any         |Seed for random number generator.|
|ECHO        |0        |0,1         |If `1`, detailed information will be printed on terminal window.|
|TIME_UNIT   |1e-6     |any         |Discrete time step to be used for spike conversion. Define as unit of seconds.|
|PLOT_RASTER |1        |0,1         |If `1`, raster plots will be generated.|
|PLOT_WFORM  |1        |0,1         |If `1`, waveform plots will be generated.|
|PLOT_NUMBER |16       |any         |Number of data for the raster and waveform plottings.|
|PLOT_INDEX_FROM|0     |any         |Data index number for plotting. The index range will be `[PLOT_INDEX_FROM : PLOT_INDEX_FROM + PLOT_NUMBER]`.|
|PLOT_INDEX_LIST|None  |any         |List of data index number for plotting, such as "5 9 14". This is for defining arbitrary uncontinued index numbers. If this is defined, `PLOT_INDEX_FROM` and `PLOT_NUMBER` will be ignored.|
|PLOT_TITLE  |1        |0,1         |If `1`, .wav file name is added as title.|
|PICKLE_FILE |train.pickle|any      |Pickle file name for plotting and analysis. Default is `OUTPUT_FILES[0]` with replacing the suffix `.bin` to `.pickle`. (This should be one of `OUTPUT_FILES` by replacing the suffix:.bin to .pickle)|
|WINDOW_SIZE |100000   |any         |The size of moving window for analysis, as unit of micro seconds|
|WINDOW_STRIDE|50000   |any         |The size of moving window for analysis, as unit of micro seconds|
|VCSV_FILE_PREFIX|c    |any         |VCSV file name prefix for default `VCSV_FILE_LIST`. (see `VCSV_FILE_LIST` in Major options)|
|VCSV_FILE_SUFFIX|.vcsv|any         |VCSV file name suffix for default `VCSV_FILE_LIST`. (see `VCSV_FILE_LIST` in Major options)|
|PREEMPHASIS_COEF|0.97 |any         |Coefficient paramter for preemphasis. valid only when `PREEMPHASIS=1`|

## Categories
Total number of .wav files are 105835. Number of .wav files listed in `testing_list.txt` and `validation_list.txt` are 11005 and 9981, respectively. The .wav files listed in `testing_list.txt` and `validation_list.txt` are not overlapped.

### Core 20 words

|Label|Category | Number of data (.wav file) all|in testing_list.txt|in validation_list.txt|
|:-----------|:--------|:-----------|:-------|:--------|
|0|zero| 4052|  418| 384|
|1|one| 3890|   399| 351|
|2|two| 3880|   424| 345|
|3|three| 3727| 405| 356|
|4|four| 3728|  400| 373|
|5|five| 4052|  445| 367|
|6|six| 3860|   394| 378|
|7|seven| 3998| 406| 387|
|8|eight| 3787| 408| 346|
|9|nine| 3934|  408| 356|
|10|yes| 4044|  419| 397|
|11|no| 3941|   405| 406|
|12|up| 3723|   425| 350|
|13|down| 3917| 406| 377|
|14|left| 3801| 412| 352|
|15|right| 3778|396| 363|
|16|on| 3845|   396| 363|
|17|off| 3745|  402| 373|
|18|stop| 3872| 411| 350|
|19|go| 3880|   402| 372|

### The words to help distinguish unrecognized words

|Label|Category | Number of data (.wav file) all|in testing_list.txt|in validation_list.txt|
|:----|:--------|:-------|:-------|:------------|
|20|cat| 2031|     194| 180|
|21|dog| 2128|     220| 197|
|22|bird| 2064|    185| 182|
|23|house| 2113|   191| 195|
|24|bed| 2014|     207| 213|
|25|tree| 1759|    193| 159|
|26|happy| 2054|   203| 219|
|27|wow| 2123|     206| 193|
|28|sheila| 2022|  212| 204|
|29|marvin| 2100|  195| 195|
|30|forward| 1557| 155| 146|
|31|backward| 1664|165| 153|
|32|follow| 1579|  172| 132|
|33|learn| 1575|   161| 128|
|34|visual| 1592|  165| 139|
|35|_background_noise_| 6| | |
