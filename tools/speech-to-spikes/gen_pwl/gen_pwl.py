# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

if __name__ == '__main__':
    import argparse
    from gscd_ext import AudioCore

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_file', help='audio file name')
    parser.add_argument('output_file', help='output file name')
    parser.add_argument('--vscale', help='voltage scale factor', type=float, default=0.0894e-3)
    parser.add_argument('--vshift', help='voltage shift factor', type=float, default=1.0)
    parser.add_argument('--echo', help='echo information', action='store_true', default=False)
    args = parser.parse_args()

    sound = AudioCore(args.audio_file)
    if args.echo:
        sound.echo_params()
    sound.dump_as_pwl(args.output_file, scale = args.vscale, shift = args.vshift, echo = args.echo)
