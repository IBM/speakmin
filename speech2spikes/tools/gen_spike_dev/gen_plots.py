
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse
    import numpy as np
    import os
    from gscd_ext import AudioCore

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_file', help='audio file name')
    parser.add_argument('--n_mels', help='number of mel frequency bin', type=int, default=16)
    parser.add_argument('--alpha', help='potential scaling coefficient', type=float, default=1.0)
    parser.add_argument('--vth', help='spike threshold', type=float, default=0.02)
    parser.add_argument('--echo', help='echo messages', action='store_true', default=False)
    parser.add_argument('--output_path', help='output path name', default='.')
    parser.add_argument('--align', help='align sound data', action='store_true', default=False)
    parser.add_argument('--time_unit', help='time unit', type=float, default=1)
    args = parser.parse_args()

    sound = AudioCore(args.audio_file)

    #
    # Original data plot
    #-------------------------------
    ofile = os.path.join(args.output_path, 'data_original.png')
    fig, ax = plt.subplots(figsize = (8, 6))
    sound.plot_original(ax)
    ax.set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(ofile)
    plt.close(fig)

    if args.align:
        sound.align_sound(index_align = 8000, n_points = 16000, pad_zero = False)

        ofile = os.path.join(args.output_path, 'data_aligned.png')
        fig, ax = plt.subplots(figsize = (8, 6))
        sound.plot_original(ax)
        ax.set_xlabel('Time [s]')
        fig.tight_layout()
        fig.savefig(ofile)
        plt.close(fig)

    #
    # Mel frequency banks
    #-------------------------------
    ofile = os.path.join(args.output_path, 'mel_freq_filter_banks.png')
    fig, ax = plt.subplots(figsize = (8, 6))
    sound.plot_mel_filter_banks(ax, args.n_mels, echo=args.echo) # librosa (Slaney's)
    ax.set_xlabel('Frequency [Hz]')
    fig.tight_layout()
    fig.savefig(ofile)
    plt.close(fig)

    #freq_max = 8000y
    #norm = True
    #style = 'slaney'
    #sound.mel_freq_banks(args.n_mels, freq_max, style=style, norm=norm)

    #
    # FFT original
    #-------------------------------
    ofile = os.path.join(args.output_path, 'dft_original.png')
    fig, ax = plt.subplots(figsize = (8, 6))
    sound.plot_dft(ax)
    ax.set_xlabel('Frequency [Hz]')
    fig.tight_layout()
    fig.savefig(ofile)
    plt.close(fig)

    #
    # Mel freqeuncy filtered
    #-------------------------------
    ofile1 = os.path.join(args.output_path, 'mel_freq_filtered.png')
    ofile2 = os.path.join(args.output_path, 'integrate_and_fire.png')
    ofile3 = os.path.join(args.output_path, 'raster_plots.png')
    ofile4 = os.path.join(args.output_path, 'raster_plots_time_unit.png')
    fig1, ax1 = plt.subplots(4, 4, figsize = (16, 12))
    fig2, ax2 = plt.subplots(4, 4, figsize = (16, 12))
    fig3, ax3 = plt.subplots(figsize = (8, 6))
    fig4, ax4 = plt.subplots(figsize = (8, 6))

    ax3.grid()
    ax4.grid()
    tdiff_min_list = np.zeros(args.n_mels)
    all_max_mag = 0
    for ch in range(args.n_mels):
        sig_filtered = sound.bpf_melfreq(args.n_mels, ch)
        sig_time = sound.time
        row, col = ch // 4, ch % 4

        #-- fig1
        ax1[row, col].plot(sig_time, sig_filtered)
        max_mag = max(abs(sig_filtered))
        if all_max_mag < max_mag:
            all_max_mag = max_mag

        #-- fig2
        potential_t, potential_v, time_spike = sound.integrate_and_fire(sig_filtered, args.vth, alpha=args.alpha)
        ax2[row, col].plot(potential_t, potential_v)

        #-- fig3
        if len(time_spike) > 0:
            ax3.plot(time_spike, [ch] * len(time_spike), 'o')
            if args.time_unit != 1.0:
                time_in_unit = [int(round(time/args.time_unit)) for time in time_spike]
                ax4.plot(time_in_unit, [ch] * len(time_spike), 'o')

            tdiff_min = time_spike[-1] # large value
            for index, time in enumerate(time_spike):
                if index != 0:
                    tdiff = time_spike[index] - time_spike[index - 1]
                    if tdiff < tdiff_min:
                        tdiff_min = tdiff
        else:
            tdiff_min = float('inf')
        tdiff_min_list[ch] = tdiff_min

    if args.echo:
        print(f'ch = {np.argmin(tdiff_min_list)}, tdiff_min = {np.min(tdiff_min_list)}')

    for ch in range(args.n_mels):
        row, col = ch // 4, ch % 4
        ax1[row, col].set_ylim(-all_max_mag, all_max_mag)

    fig1.tight_layout()
    fig1.savefig(ofile1)
    plt.close(fig1)

    fig2.tight_layout()
    fig2.savefig(ofile2)
    plt.close(fig2)

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Channel#')
    ax3.set_ylim(-0.5, args.n_mels - 0.5)
    ax3.set_xlim(-0.05, 1.05)
    fig3.tight_layout()
    fig3.savefig(ofile3)
    plt.close(fig3)

    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Channel#')
    ax4.set_ylim(-0.5, args.n_mels - 0.5)
    ax4.set_xlim(-0.05 / args.time_unit, 1.05 / args.time_unit)
    fig4.tight_layout()
    fig4.savefig(ofile4)
    plt.close(fig4)
