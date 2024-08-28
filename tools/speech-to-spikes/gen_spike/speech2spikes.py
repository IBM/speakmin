# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import pickle
    import os
    from gscd_ext import GSCD
    from gscd_ext import AudioCore

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_file', nargs='*', help='output spike file name (.bin)')
    parser.add_argument('--dataset_path', help='path for Google speech command dataset',
                        default='../../dataset/speech_commands_v0.02')
    parser.add_argument('--category', help='list of categories', nargs='*', default=['zero', 'one', 'two'])
    parser.add_argument('--split_number', help='list of split numbers', nargs='*', type=int, default=[16, 16])
    parser.add_argument('--n_mels', help='number of mel frequency bin', type=int, default=16)
    parser.add_argument('--alpha', help='potential scaling coefficient', type=float, default=1.0)
    parser.add_argument('--vth', help='spike threshold', type=float, default=0.002)
    parser.add_argument('--time_unit', help='time unit', type=float, default=1e-6)
    parser.add_argument('--align', help='align sound data', action='store_true', default=False)
    parser.add_argument('--rng_seed', help='seed for random number generator', type=int, default=10)
    parser.add_argument('--echo', help='echo messages', action='store_true', default=False)
    parser.add_argument('--norm', help='normalize volume', action='store_true', default=False)
    parser.add_argument('--norm_target_dBFS', help='peak in dB for normalization (valid only with --norm)', type=float, default=-31.782)
    parser.add_argument('--wav_file_source', help='.wav file selection from which pool',
                        choices=['all', 'testing', 'validation', 'not_testing', 'not_validation', 'not_testing_validation'], default='all')
    parser.add_argument('--use_all_in_source', help='use all .wav files in source. split_number is ignored.', action='store_true', default=False)
    parser.add_argument('--mel_norm', help='normalization method for librosa mel frequency bank', default='slaney')
    parser.add_argument('--num_process', help='number of process for multithreading', type=int, default=8)
    parser.add_argument('--vcsv_file_list', help='list of vcs files (define if use vcs files)', nargs='*', default=None)
    parser.add_argument('--leak_enable', help='enable leak function in IF neuron', action='store_true', default=False)
    parser.add_argument('--leak_tau', help='leak time constant for IF neurons', type=float, default=16000e-6)
    parser.add_argument('--preemphasis', help='enable preemphasis', action='store_true', default=False)
    parser.add_argument('--preemphasis_coef', help='preemphasis coefficient number. valid only with --preemphasis', type=float, default=0.97)
    args = parser.parse_args()
    if args.use_all_in_source:
        output_file_list = [args.output_file[0]]
        split_number_list = [128] # dummy. this is overwritten in random_split()
    else:
        output_file_list = args.output_file
        split_number_list = args.split_number

    assert len(output_file_list) == len(split_number_list), \
        f'len(output_file_list)({len(output_file_list)}) and len(split_number_list)({len(split_number_list)}) should be equal.'

    if args.vcsv_file_list != None:
        assert len(args.vcsv_file_list) == args.n_mels, \
            f'number of vcsv_file_list({len(args.vcsv_file_list)}) and n_mels({args.n_mels}) are not matched.'

    mel_norm = args.mel_norm
    if mel_norm.upper() == 'NONE':
        mel_norm = None
    pad_zero = True

    dataset = GSCD(args.dataset_path)
    if args.echo:
        dataset.show_categories()

    #
    # Convert
    #------------------------------
    splitted_wav_files = dataset.random_split(args.category, split_number_list, rng = None, rng_seed = args.rng_seed,
                                              wav_file_source = args.wav_file_source, use_all_in_source = args.use_all_in_source)

    spikes_list_of_dict = dataset.convert2spikes(splitted_wav_files, args.n_mels, args.vth,
                                                 align = args.align, alpha = args.alpha, time_unit = args.time_unit,
                                                 pad_zero = pad_zero, norm = args.norm, norm_target_dBFS = args.norm_target_dBFS,
                                                 mel_norm = mel_norm, num_process = args.num_process,
                                                 vcsv_file_list = args.vcsv_file_list,
                                                 leak_enable = args.leak_enable, leak_tau = args.leak_tau,
                                                 preemphasis = args.preemphasis, preemphasis_coef = args.preemphasis_coef)

    #
    # Dump to file
    #------------------------------
    splitted_spikes_list_of_dict = dataset.split_and_shuffle_for_dump(spikes_list_of_dict, shuffle=True, rand_seed = args.rng_seed)

    for spikes_list_of_dict_for_dump, output_file in zip(splitted_spikes_list_of_dict, output_file_list):
        dataset.dump_as_binary(spikes_list_of_dict_for_dump, output_file)

        pickle_file = os.path.splitext(output_file)[0] + '.pickle'
        dataset.dump_as_pickle(spikes_list_of_dict_for_dump, pickle_file)

    #
    # Plot mel-frequency bank
    #------------------------------
    fig, ax = plt.subplots(figsize = (8, 6))
    if args.vcsv_file_list != None:
        for csv_file in args.vcsv_file_list:
            freq, melfb = AudioCore.load_vcsv(csv_file)
            ax.plot(freq, [abs(complex_val) for complex_val in melfb])
    else:
        sr = 16000 # 62.5 us
        n_fft = 16000
        AudioCore.plot_librosa_melfreq_bfp(ax, n_mels = args.n_mels, sr = sr, n_fft = n_fft, norm = mel_norm)
    ax.set_xlabel('Frequency [Hz]')
    fig.tight_layout()
    fig.savefig('mel_freq_bank.png')
    ax.set_xscale('log')
    fig.savefig('mel_freq_bank_log.png')
