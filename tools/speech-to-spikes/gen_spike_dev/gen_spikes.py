# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

if __name__ == '__main__':
    import argparse
    import pickle
    import numpy as np
    from gscd_ext import AudioCore

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_file', help='audio file name')
    parser.add_argument('output_file', help='output spike file name')
    parser.add_argument('--n_mels', help='number of mel frequency bin', type=int, default=16)
    parser.add_argument('--alpha', help='potential scaling coefficient', type=float, default=1.0)
    parser.add_argument('--vth', help='spike threshold', type=float, default=0.02)
    parser.add_argument('--echo', help='echo messages', action='store_true', default=False)
    parser.add_argument('--align', help='align sound data', action='store_true', default=False)
    parser.add_argument('--time_unit', help='time unit', type=float, default=1)
    args = parser.parse_args()

    sound = AudioCore(args.audio_file)
    if args.align:
        sound.align_sound(index_align = 8000, n_points = 16000, pad_zero = False)
    spikes_sorted = sound.speech2spikes(args.n_mels, args.vth, alpha = args.alpha, time_unit = args.time_unit)

    if args.echo:
        print(f'spikes_sorted = {spikes_sorted}')

    with open(args.output_file, 'wb') as f:
        pickle.dump(spikes_sorted, f)
