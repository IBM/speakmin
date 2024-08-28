# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

if __name__ == '__main__':
    import argparse
    import pickle
    import os
    from gscd_ext import GSCD

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pickle_file', help='spike pickle file (.pickle)')
    parser.add_argument('--plot_raster', help='generate raster plots', action='store_true', default=True)
    parser.add_argument('--plot_wform', help='generate waveform plots', action='store_true', default=True)
    parser.add_argument('--plot_number', help='number of plots', type=int, default=16)
    parser.add_argument('--plot_index_from', help='plot index from', type=int, default=0)
    parser.add_argument('--plot_index_list', help='define plot index as list', nargs='*', type=int, default=[None])
    parser.add_argument('--plot_title', help='add .wav file name as title', action='store_true', default=False)
    args = parser.parse_args()

    #
    # Load data
    #-------------------------
    with open(args.pickle_file, 'rb') as f:
        spikes_list_of_dict = pickle.load(f)

    #
    # Plot data index
    #-------------------------
    if args.plot_index_list[0] == None:
        data_index_list = list(range(args.plot_index_from, args.plot_index_from + args.plot_number))
    else:
        data_index_list = args.plot_index_list

    #
    # Plot raster
    #-------------------------
    if args.plot_raster:
        GSCD.raster_plots(spikes_list_of_dict, data_index_list = data_index_list, plot_title = args.plot_title)

    #
    # Plot waveform
    #-------------------------
    if args.plot_wform:
        align = spikes_list_of_dict[0]['align']
        norm = spikes_list_of_dict[0]['norm']
        preemphasis = spikes_list_of_dict[0]['preemphasis']

        # plot original data
        if align or norm or preemphasis:
            ylim_max = GSCD.wform_lookup_ylim(spikes_list_of_dict, data_index_list = data_index_list,
                                              plot_original = True)
            GSCD.wform_plots(spikes_list_of_dict, data_index_list = data_index_list,
                             ylim = [-ylim_max, ylim_max], plot_title = args.plot_title, plot_original = True)
        # plot
        ylim_max = GSCD.wform_lookup_ylim(spikes_list_of_dict, data_index_list = data_index_list)
        GSCD.wform_plots(spikes_list_of_dict, data_index_list = data_index_list,
                         ylim = [-ylim_max, ylim_max], plot_title = args.plot_title)
