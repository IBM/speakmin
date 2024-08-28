# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0

import pickle
import itertools
import math
import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class Analysis(object):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.spikes_list_of_dict = pickle.load(f)
            category_list_all = [data_dict['category'] for data_dict in self.spikes_list_of_dict]
            self.categories = sorted(set(category_list_all))
            print(f'len(spikes_list_of_dict) = {len(self.spikes_list_of_dict)}')
            print(f'categories = {self.categories}')

    def compress_data_by_moving_window(self, category, window_size, stride,
                                       t_start = 0, t_end = 1000000, num_channels = 16):
        tindex_list_s = range(t_start, t_end, stride)
        tindex_list_e = range(t_start + stride, t_end + stride, stride)
        list_of_data_dict = [data_dict for data_dict in self.spikes_list_of_dict if data_dict['category'] == category]
        sorted_list_of_data_dict = sorted(list_of_data_dict, key=lambda x:x['data_index'])

        spikes_info = []
        spikes_3d = np.zeros([len(sorted_list_of_data_dict), len(tindex_list_s), num_channels], dtype=int)
        for data_index, data_dict in enumerate(sorted_list_of_data_dict):
            spikes = data_dict['spikes']
            for window_index, [tindex_s, tindex_e] in enumerate(zip(tindex_list_s, tindex_list_e)):
                spikes_in_window = [spike for spike in spikes if tindex_s <= spike[0] and spike[0] < tindex_e]
                spikes_1d = np.zeros(num_channels, dtype=int)
                if len(spikes_in_window) > 0:
                    #print(f'tindex_s, tindex_e = {tindex_s}, {tindex_e}')
                    #print(f'spikes_in_window = {spikes_in_window}')

                    for ch in range(num_channels):
                        num_spikes = sum([1 for spike in spikes_in_window if spike[1] == ch])
                        spikes_1d[ch] = num_spikes

                spikes_3d[data_index][window_index] = spikes_1d

            info_dict = data_dict.copy()
            del info_dict['spikes']
            spikes_info.append(info_dict)

        return spikes_3d, spikes_info # spikes_3d[data_index][window_index][channel_index] spikes_info[data_index]{}

    def cal_distance_among_categories(self, window_size, stride,
                                      t_start = 0, t_end = 1000000, num_channels = 16):
        dict_spikes_3d_averaged = {}
        for category in self.categories:
            spikes_3d, spikes_info = self.compress_data_by_moving_window(
                category, window_size, stride, t_start = t_start, t_end = t_end, num_channels = num_channels)
            spikes_3d_averaged = np.average(spikes_3d, axis=0)
            dict_spikes_3d_averaged[category] = np.ravel(spikes_3d_averaged) # flatten to 1-D array

        distance_among_categories = []
        for index_a, cat_a in enumerate(self.categories):
            list_1d = []
            for index_b, cat_b in enumerate(self.categories):
                distance = np.linalg.norm(dict_spikes_3d_averaged[cat_a] - dict_spikes_3d_averaged[cat_b])
                list_1d.append(distance)
            distance_among_categories.append(list_1d)

        #print(f'distance_among_categories = {distance_among_categories}')

        return distance_among_categories # 2D list

    def cal_variance(self, window_size, stride,
                     t_start = 0, t_end = 1000000, num_channels = 16):

        variance_list = []
        for category in self.categories:
            variance, sd = self.cal_variance_in_category(
                category, window_size, stride, t_start = t_start, t_end = t_end, num_channels = num_channels)
            num_of_data = sum([1 for data_dict in self.spikes_list_of_dict if data_dict['category'] == category])
            variance_list.append({'category':category, 'num_of_data':num_of_data, 'variance':variance, 'sd':sd})

        return variance_list

    def cal_variance_in_category(self, category, window_size, stride,
                                 t_start = 0, t_end = 1000000, num_channels = 16):

        spikes_3d, spikes_info = self.compress_data_by_moving_window(
            category, window_size, stride, t_start = t_start, t_end = t_end, num_channels = num_channels)
        spikes_3d_averaged = np.average(spikes_3d, axis=0)

        diff2_list = []
        for spikes_2d in spikes_3d:
            diff = np.linalg.norm(spikes_3d_averaged - spikes_2d) # cal as distance between average and target
            diff2_list.append(diff ** 2)

        variance = sum(diff2_list) / len(diff2_list)
        sd = math.sqrt(variance)
        #print(f'category={category}, variance={variance}, sd={sd}')
        return variance, sd

    #
    # Plot utilities
    #-----------------------------
    def plot_compressed_data(self, window_size, stride,
                             t_start = 0, t_end = 1000000, num_channels = 16, plot_title = True):
        averaged_spikes = {}
        for category_index, category in enumerate(self.categories):
            spikes_3d, spikes_info = self.compress_data_by_moving_window(
                category, window_size, stride, t_start = t_start, t_end = t_end, num_channels = num_channels)
            spikes_3d_averaged = np.average(spikes_3d, axis=0)
            averaged_spikes[category] = spikes_3d_averaged

            ofile = f'{category}_compressed_data.png'
            sub_figsize = [4, 3]
            fig_col = 4
            fig_row = int((len(data_index_list) + fig_col - 1) / fig_col)
            figsize = [sub_figsize[0] * fig_col, sub_figsize[1] * fig_row]
            fig, ax = plt.subplots(fig_row, fig_col, figsize = figsize)

            # find max
            tmp_list = [spikes_2d for data_index, spikes_2d in enumerate(spikes_3d) if data_index in data_index_list]
            max_value = max(itertools.chain(*itertools.chain(*tmp_list)))
            for index, data_index in enumerate(data_index_list):
                transposed = list(zip(*spikes_3d[data_index]))
                row, col = index // fig_col, index % fig_col
                ax_target = ax[row, col] if fig_col < len(data_index_list) else ax[col]
                ax_target.pcolor(transposed, cmap=plt.cm.Reds, vmin=0, vmax=max_value)
                ax_target.set_xlabel('Window index')
                ax_target.set_ylabel('Cnannel#')
                if plot_title:
                    wav_file = os.path.basename(spikes_info[data_index]['wav_file'])
                    ax_target.set_title(f'"{category}",{data_index},{wav_file}')

            fig.tight_layout()
            fig.savefig(ofile)
            plt.close(fig)

        av_ofile = f'averaged_compressed_data.png'
        av_sub_figsize = [4, 3]
        av_fig_col = 4
        av_fig_row = int((len(spikes.categories) + av_fig_col - 1) / av_fig_col)
        av_figsize = [av_sub_figsize[0] * av_fig_col, av_sub_figsize[1] * av_fig_row]
        av_fig, av_ax = plt.subplots(av_fig_row, av_fig_col, figsize = av_figsize)
        # find max
        av_max_value = max(itertools.chain(*itertools.chain(*averaged_spikes.values())))
        for category_index, category in enumerate(spikes.categories):
            av_transposed = list(zip(*averaged_spikes[category]))
            av_row, av_col = category_index // av_fig_col, category_index % av_fig_col
            av_ax_target = av_ax[av_row, av_col] if av_fig_col < len(spikes.categories) else av_ax[av_col]
            av_ax_target.pcolor(av_transposed, cmap=plt.cm.Greens, vmin=0, vmax=av_max_value)
            av_ax_target.set_xlabel('Window index')
            av_ax_target.set_ylabel('Cnannel#')
            if plot_title:
                av_ax_target.set_title(f'averaged,{category}')

        av_fig.tight_layout()
        av_fig.savefig(av_ofile)
        plt.close(av_fig)

    def plot_distance_among_categories(self, ax, distance_among_categories, fmt='.1f'):
        mappable = ax.pcolor(distance_among_categories, cmap=plt.cm.Blues)
        ax.set_xticks(np.arange(len(spikes.categories)) + 0.5)
        ax.set_yticks(np.arange(len(spikes.categories)) + 0.5)
        ax.set_xticklabels(spikes.categories)
        ax.set_yticklabels(spikes.categories)
        max_value = max(list(map(lambda x: max(x), distance_among_categories)))
        for [i, j], value in np.ndenumerate(distance_among_categories):
            color = 'black' if value < (max_value / 2) else 'white'
            ax.text(j + 0.5, i + 0.5, f'{value:{fmt}}', ha='center', va='center', color=color)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pickle_file', help='spike pickle file (.pickle)')
    parser.add_argument('--window_size', help='window size of moving window', type=int, default=100000)
    parser.add_argument('--window_stride', help='stride of moving window', type=int, default=50000)
    parser.add_argument('--plot_number', help='number of plots', type=int, default=16)
    parser.add_argument('--plot_index_from', help='plot index from', type=int, default=0)
    parser.add_argument('--plot_index_list', help='define plot index as list', nargs='*', type=int, default=[None])
    args = parser.parse_args()

    spikes = Analysis(args.pickle_file)

    #
    # Fixed params
    #-------------------------
    t_start, t_end = [0, 1000000]
    num_channels = spikes.spikes_list_of_dict[0]['n_mels'] # use the first data assuming all data is same
    plot_title = True

    #
    # Plot data index
    #-------------------------
    if args.plot_index_list[0] == None:
        data_index_list = list(range(args.plot_index_from, args.plot_index_from + args.plot_number))
    else:
        data_index_list = args.plot_index_list

    #
    # Plot compressed data by
    #------------------------------
    spikes.plot_compressed_data(args.window_size, args.window_stride,
                                t_start = t_start, t_end = t_end, num_channels = num_channels, plot_title = plot_title)

    #
    # Plot class-to-class distance
    #------------------------------
    distance_among_categories = spikes.cal_distance_among_categories(
        args.window_size, args.window_stride, t_start = t_start, t_end = t_end, num_channels = num_channels)

    figsize_base = [0.6, 0.45]
    figsize = [value * len(spikes.categories) for value in figsize_base]
    ofile = 'c2c_distance.png'
    fig, ax = plt.subplots(figsize = figsize)
    spikes.plot_distance_among_categories(ax, distance_among_categories)
    #plt.colorbar(mappable, ax=ax)
    fig.tight_layout()
    fig.savefig(ofile)
    plt.close(fig)

    #
    # Variance/Standard deviation table
    #------------------------------
    #print(spikes.categories)
    variance_list = spikes.cal_variance(args.window_size, args.window_stride)
    d = {}
    d['category'] = [data_dict['category'] for data_dict in variance_list]
    d['number of data'] = [data_dict['num_of_data'] for data_dict in variance_list]
    d['standard deviation'] = [f'{data_dict["sd"]:.2f}' for data_dict in variance_list]
    df = pd.DataFrame(d)
    ofile = 'class_sd.md'
    with open(ofile, 'w') as f:
        f.write(df.to_markdown(index=False))
    print(df.to_markdown(index=False))

    #
    # Plot Class-to-class distance / Standard deviation as an index for separate-ability
    #------------------------------
    sd = [data_dict["sd"] for data_dict in variance_list]

    distance_sd = []
    for row_index, [row_distance_among_categories, row_sd] in enumerate(zip(distance_among_categories, sd)):
        #print(f'row_index={row_index}, row_sd={row_sd}')
        #print(f'{row_distance_among_categories}')
        row_distance_sd = [distance / row_sd for distance in row_distance_among_categories]
        col_distance_sd = [distance / col_sd for distance, col_sd in zip(row_distance_among_categories, sd)]
        #print(f'{row_distance_sd}')
        #print(f'{col_distance_sd}')
        averaged_distance_sd = [(row_val + col_val) / 2 for row_val, col_val in zip(row_distance_sd, col_distance_sd)]
        #print(f'{averaged_distance_sd}')
        distance_sd.append(averaged_distance_sd)

    num_categories = len(spikes.categories)
    total_averaged_distance_sd = sum(list(itertools.chain(*distance_sd))) / (num_categories * num_categories - num_categories)

    figsize_base = [0.6, 0.45]
    figsize = [value * num_categories for value in figsize_base]
    ofile = 'c2c_distance_sd.png'
    fig, ax = plt.subplots(figsize = figsize)
    spikes.plot_distance_among_categories(ax, distance_sd, fmt='.2f')
    #plt.colorbar(mappable, ax=ax)
    ax.set_title(f'average={total_averaged_distance_sd:.2f}')
    fig.tight_layout()
    fig.savefig(ofile)
    plt.close(fig)
