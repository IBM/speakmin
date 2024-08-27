# (C) Copyright IBM Corp. 2024
#
# Python class for Google Speech Command Dataset (GSCD)
#
#
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import pickle
import random
import multiprocessing
from .audio_core import AudioCore

class GSCD(object):
    def __init__(self, dataset_path):
        assert os.path.isdir(dataset_path)
        self.dataset_path = dataset_path
        self.categories = [os.path.basename(path) for path in glob.glob(dataset_path + '/*') if os.path.isdir(path)]

        self.wav_files = []
        self.uid = {}
        uid = 0 # unique id for each .wav file
        for category in self.categories:
            wav_files = glob.glob(os.path.join(dataset_path, category, '*.wav'))
            for wav_file in wav_files:
                self.wav_files.append({'uid':uid, 'category':category, 'wav_file':wav_file})
                self.uid[wav_file] = uid
                uid +=  1

        self.queue = multiprocessing.Manager().Queue()
        self.result_queue = multiprocessing.Manager().Queue()

    def _get_wav_file_names_from_file(self, list_file):
        with open(list_file, 'r') as f:
            list_of_dict = []
            for s_line in f:
                category, wav_file = s_line.strip().split('/')
                wav_file_full_path = os.path.join(self.dataset_path, category, wav_file)
                uid = self.uid[wav_file_full_path]
                list_of_dict.append({'uid':uid, 'category':category, 'wav_file':wav_file_full_path})

        categories = set([d.get('category') for d in list_of_dict])

        return list_of_dict, categories

    @classmethod
    def category2index(cls, category):
        category_list = [
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'yes', 'no', 'up', 'down',
            'left', 'right', 'on', 'off', 'stop', 'go', 'cat', 'dog', 'bird', 'house', 'bed', 'tree', 'happy', 'wow',
            'sheila', 'marvin', 'forward', 'backward', 'follow', 'learn', 'visual', '_background_noise_'
        ]
        assert category in category_list
        return category_list.index(category)

    def show_categories(self):
        for category in self.categories:
            print(f'{category}: {len(self.wav_files[category])}')

    def _exclude_wav_file(self, list_of_dict, exclude_list_of_dict, dict_key = 'wav_file'):
        exclude_wav_file_list = [dict_data[dict_key] for dict_data in exclude_list_of_dict]
        new_list_of_dict = []
        for dict_data in list_of_dict:
            if dict_data[dict_key] not in exclude_wav_file_list:
                new_list_of_dict.append(dict_data)
        return new_list_of_dict

    def _select_wav_file_source(self, wav_file_source):
        assert wav_file_source in ['all', 'testing', 'validation', 'not_testing', 'not_validation', 'not_testing_validation']

        if wav_file_source == 'all':
            source_wav_files, source_categories = [self.wav_files, self.categories]
        elif wav_file_source == 'testing':
            source_wav_files, source_categories = self._get_wav_file_names_from_file(os.path.join(self.dataset_path, 'testing_list.txt'))
        elif wav_file_source == 'validation':
            source_wav_files, source_categories = self._get_wav_file_names_from_file(os.path.join(self.dataset_path, 'validation_list.txt'))
        elif wav_file_source == 'not_testing':
            tst_source_wav_files, tst_source_categories = self._get_wav_file_names_from_file(os.path.join(self.dataset_path, 'testing_list.txt'))
            source_wav_files = self._exclude_wav_file(self.wav_files, tst_source_wav_files)
            source_categories = set([d.get('category') for d in source_wav_files])
        elif wav_file_source == 'not_validation':
            val_source_wav_files, val_source_categories = self._get_wav_file_names_from_file(os.path.join(self.dataset_path, 'validation_list.txt'))
            source_wav_files = self._exclude_wav_file(self.wav_files, val_source_wav_files)
            source_categories = set([d.get('category') for d in source_wav_files])
        elif wav_file_source == 'not_testing_validation':
            tst_source_wav_files, tst_source_categories = self._get_wav_file_names_from_file(os.path.join(self.dataset_path, 'testing_list.txt'))
            val_source_wav_files, val_source_categories = self._get_wav_file_names_from_file(os.path.join(self.dataset_path, 'validation_list.txt'))
            not_tst_source_wav_files = self._exclude_wav_file(self.wav_files, tst_source_wav_files)
            source_wav_files = self._exclude_wav_file(not_tst_source_wav_files, val_source_wav_files)
            source_categories = set([d.get('category') for d in source_wav_files])

        return source_wav_files, source_categories

    def random_split(self, category_list, split_number_list, rng = None, rng_seed = 10, wav_file_source = 'all', use_all_in_source = False):
        '''
        category_list: List of GSCD categories, such as ['one', 'two', 'three']
        split_number_list: List of numbers to be splitted, such as [100, 100]
        rng: numpy RandomState object (if not provided, internally generates new one with seed of rng_seed
        rng_seed: Valid only if rng == None
        wav_file_source: source of .wav file. either of 'all', 'testing', 'validation', 'not_testing', 'not_valiation', or 'not_testing_validation'.
        use_all_in_source: if True, all .wav files in wav_file_source will be used. split_number_list is ignored.
        '''
        source_wav_files, source_categories = self._select_wav_file_source(wav_file_source)

        assert all([category in source_categories for category in category_list])

        if rng == None:
            rng = np.random.RandomState(rng_seed)

        splitted_wav_files = []
        for category in category_list:
            wav_files = [dict_data for dict_data in source_wav_files if dict_data['category'] == category]
            if use_all_in_source:
                split_number_list = [len(wav_files)]
            assert sum(split_number_list) <= len(wav_files)
            wav_files_shuffled = rng.permutation(wav_files)
            index_from = 0
            for split_index, split_number in enumerate(split_number_list):
                index_to = index_from + split_number
                for data_index, dict_data in enumerate(wav_files_shuffled[index_from : index_to]):
                    dict_data['split_index'] = split_index
                    dict_data['data_index'] = data_index
                    dict_data['label'] = GSCD.category2index(category)
                    splitted_wav_files.append(dict_data)
                index_from = index_to

        return splitted_wav_files    # list of dict

    def convert2spikes(self, splitted_wav_files, n_mels, vth, align = False, alpha = 1.0, time_unit = 1.0,
                       pad_zero = True, norm = False, norm_target_dBFS = -0.1, mel_norm = 'slaney', num_process = 4,
                       vcsv_file_list = None, leak_enable = False, leak_tau = 16000e-6,
                       preemphasis = False, preemphasis_coef = 0.97):

        self._put_items_in_queue(splitted_wav_files, n_mels, vth, align = align, alpha = alpha, time_unit = time_unit,
                                 pad_zero = pad_zero, norm = norm, norm_target_dBFS = norm_target_dBFS, mel_norm = mel_norm, num_process = num_process,
                                 vcsv_file_list = vcsv_file_list, leak_enable = leak_enable, leak_tau = leak_tau,
                                 preemphasis = preemphasis, preemphasis_coef = preemphasis_coef)

        self._start_multi_process(num_process)
        spikes_list_of_dict = self._get_results_from_result_queue()

        return spikes_list_of_dict

    #
    # Queue
    #----------------------
    def _put_items_in_queue(self, splitted_wav_files, n_mels, vth, align = False, alpha = 1.0, time_unit = 1.0,
                            pad_zero = True, norm = False, norm_target_dBFS = -0.1, mel_norm = 'slaney', num_process = 4,
                            vcsv_file_list = None, leak_enable = False, leak_tau = 16000e-6,
                            preemphasis = False, preemphasis_coef = 0.97):

        for dict_data in splitted_wav_files:
            keys = ['n_mels', 'vth', 'align', 'alpha', 'time_unit', 'pad_zero', 'norm', 'norm_target_dBFS', 'mel_norm', 'vcsv_file_list']
            vals = [ n_mels ,  vth ,  align ,  alpha ,  time_unit ,  pad_zero ,  norm ,  norm_target_dBFS ,  mel_norm ,  vcsv_file_list ]
            keys.extend(['leak_enable', 'leak_tau', 'preemphasis', 'preemphasis_coef'])
            vals.extend([ leak_enable ,  leak_tau ,  preemphasis ,  preemphasis_coef ])
            dict_data.update(zip(keys,vals))
            self.queue.put(dict_data)

        for index in range(num_process):
            self.queue.put(None) # as ending marker

    def _start_multi_process(self, num_process):
        self.process_list = []
        for index in range(num_process):
            self.process_list.append(
                multiprocessing.Process(target = GSCD._worker_queue, args=[self.queue, self.result_queue]))
            self.process_list[index].damemon = True
            self.process_list[index].start()

        for index in range(num_process):
            self.process_list[index].join()

    def _get_results_from_result_queue(self):
        qsize = self.result_queue.qsize()
        results = [None] * qsize
        for item_index in range(qsize):
            results[item_index] = self.result_queue.get()

        return results

    @classmethod
    def _worker_queue(cls, queue, result_queue):
        while True:
            item = queue.get()
            if item is None: # assumed None was put in Queue as ending marker
                break
            # Job
            sound = AudioCore(item['wav_file'])
            if item['preemphasis']:
                sound.preemphasis(coef = item['preemphasis_coef'])
            if item['norm']:
                sound.normalize(target_dBFS = item['norm_target_dBFS'])
            if item['align']:
                sound.align_sound(index_align = 8000, n_points = 16000, pad_zero = item['pad_zero'])
            spikes = sound.speech2spikes(item['n_mels'], item['vth'], alpha = item['alpha'],
                                         time_unit = item['time_unit'], norm = item['mel_norm'],
                                         vcsv_file_list = item['vcsv_file_list'],
                                         leak_enable = item['leak_enable'], leak_tau = item['leak_tau'])
            result_item = item.copy()
            result_item['num_points'] = len(spikes)
            result_item['spikes'] = spikes
            result_queue.put(result_item)

    #
    # Dump
    #-------------------------
    def split_and_shuffle_for_dump(self, spikes_list_of_dict, shuffle = True, rand_seed = 10):
        split_index_list = sorted(set([dict_item['split_index'] for dict_item in spikes_list_of_dict]))
        #print(f'split_index_list = {split_index_list}')

        splitted_spikes_list_of_dict = []
        for split_index in split_index_list:
            data_in_split = [data for data in spikes_list_of_dict if data['split_index'] == split_index]
            if shuffle:
                random.seed(rand_seed)
                random.shuffle(data_in_split)
                #data_in_split_shuffled = random.sample(data_in_split, len(data_in_split))

            # store for return
            splitted_spikes_list_of_dict.append(data_in_split)

        return splitted_spikes_list_of_dict

    def dump_as_pickle(self, spikes_list_of_dict, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(spikes_list_of_dict, f)

    def dump_as_binary(self, spikes_list_of_dict, output_file):
        # dump as binary data
        # Header: BHIBB
        # Contents: (IH) * (number of spikes)
        with open(output_file, 'wb') as f:
            for data in spikes_list_of_dict:
                # struct.pack format
                # B: unsigned char,  1 byte  (8 bit)
                # H: unsigned short, 2 bytes (16 bit)
                # I: unsigned int,   4 bytes (32 bits)
                # endian
                # <: little endian
                # >: big endian

                packed_data_head = struct.pack('>BHIIBB',data['label'], data['data_index'], data['uid'], data['num_points'], 0, 0)
                f.write(packed_data_head)
                for spike in data['spikes']:
                    packed_spike = struct.pack('>IH', spike[0], spike[1]) # time, neuron index
                    f.write(packed_spike)

    #
    # Plot utilities
    #----------------------------
    @classmethod
    def raster_plots(cls, spikes_list_of_dict, data_index_list = list(range(8)), plot_title = False):
        assert len(set([spikes_dict['split_index'] for spikes_dict in spikes_list_of_dict])) == 1, \
            f'split_index should be all same.'
        categories = set([dict_data['category'] for dict_data in spikes_list_of_dict])

        # use the first data assuming all data has same parameter value
        n_mels = spikes_list_of_dict[0]['n_mels']
        align = spikes_list_of_dict[0]['align']
        time_unit = spikes_list_of_dict[0]['time_unit']
        norm = spikes_list_of_dict[0]['norm']
        preemphasis = spikes_list_of_dict[0]['preemphasis']

        file_ext_norm = '_norm' if norm else ''
        file_ext_align = '_align' if align else ''
        file_ext_preemphasis = '_preemphasis' if preemphasis else ''
        file_ext = f'_raster{file_ext_preemphasis}{file_ext_norm}{file_ext_align}.png'
        for category in categories:
            ofile = category + file_ext
            sub_figsize = [4, 3]
            fig_col = 4
            fig_row = int((len(data_index_list) + fig_col - 1) / fig_col)
            figsize = [sub_figsize[0] * fig_col, sub_figsize[1] * fig_row]
            fig, ax = plt.subplots(fig_row, fig_col, figsize = figsize)

            target_list_of_dict = [dict_data for dict_data in spikes_list_of_dict if dict_data['category'] == category]

            sorted_target_list_of_dict = sorted(target_list_of_dict, key=lambda x:x['data_index'])
            assert all([index in [dict_data['data_index'] for dict_data in sorted_target_list_of_dict] for index in data_index_list])
            for index, data_index in enumerate(data_index_list):
                dict_data_list = [dict_tmp for dict_tmp in sorted_target_list_of_dict if dict_tmp['data_index'] == data_index]
                assert len(dict_data_list) == 1
                dict_data = dict_data_list[0]
                row, col = index // fig_col, index % fig_col
                x_list = [spike[0] * time_unit for spike in dict_data['spikes']]
                y_list = [spike[1] for spike in dict_data['spikes']]
                ax_target = ax[row, col] if fig_col < len(data_index_list) else ax[col]
                ax_target.plot(x_list, y_list, '|b')
                ax_target.set_ylim(-0.5, n_mels - 0.5)
                ax_target.set_xlim(-0.05, 1.05)
                ax_target.set_xlabel('Time [s]')
                ax_target.set_ylabel('Cnannel#')
                if plot_title:
                    ax_target.set_title(f'"{category}",{data_index},{os.path.basename(dict_data["wav_file"])}')
            fig.tight_layout()
            fig.savefig(ofile)
            plt.close(fig)

    @classmethod
    def wform_lookup_ylim(cls, spikes_list_of_dict, data_index_list = list(range(8)), plot_original = False):
        assert len(set([spikes_dict['split_index'] for spikes_dict in spikes_list_of_dict])) == 1, \
            f'split_index should be all same.'

        # use the first data assuming all data has same parameter value
        norm = False if plot_original else spikes_list_of_dict[0]['norm']
        norm_target_dBFS = spikes_list_of_dict[0]['norm_target_dBFS']
        preemphasis = False if plot_original else spikes_list_of_dict[0]['preemphasis']
        preemphasis_coef = spikes_list_of_dict[0]['preemphasis_coef']

        max_value = 0
        categories = set([dict_data['category'] for dict_data in spikes_list_of_dict])
        for category in categories:
            target_list_of_dict = [dict_data for dict_data in spikes_list_of_dict if dict_data['category'] == category]
            sorted_target_list_of_dict = sorted(target_list_of_dict, key=lambda x:x['data_index'])
            for index, data_index in enumerate(data_index_list):
                dict_data_list = [dict_tmp for dict_tmp in sorted_target_list_of_dict if dict_tmp['data_index'] == data_index]
                assert len(dict_data_list) == 1
                dict_data = dict_data_list[0]
                sound = AudioCore(dict_data['wav_file'])
                if preemphasis:
                    sound.preemphasis(coef = preemphasis_coef)
                if norm:
                    sound.normalize(target_dBFS = norm_target_dBFS)
                sig_max = max(max(sound.sig), abs(min(sound.sig)))
                if sig_max > max_value:
                    max_value = sig_max
        return max_value

    @classmethod
    def wform_plots(cls, spikes_list_of_dict, data_index_list = list(range(8)),
                    ylim = [-30000, 30000], xlim = [-0.05, 1.05], plot_title = False, plot_original = False):
        assert len(set([spikes_dict['split_index'] for spikes_dict in spikes_list_of_dict])) == 1, \
            f'split_index should be all same.'

        # use the first data assuming all data has same parameter value
        if plot_original:
            align, pad_zero, norm, preemphasis = [False, False, False, False]
        else:
            align = spikes_list_of_dict[0]['align']
            pad_zero = spikes_list_of_dict[0]['pad_zero']
            norm = spikes_list_of_dict[0]['norm']
            preemphasis = spikes_list_of_dict[0]['preemphasis']
        norm_target_dBFS = spikes_list_of_dict[0]['norm_target_dBFS']
        preemphasis_coef = spikes_list_of_dict[0]['preemphasis_coef']

        file_ext_norm = '_norm' if norm else ''
        file_ext_align = '_align' if align else ''
        file_ext_preemphasis = '_preemphasis' if preemphasis else ''
        file_ext = f'_wform{file_ext_preemphasis}{file_ext_norm}{file_ext_align}.png'

        categories = set([dict_data['category'] for dict_data in spikes_list_of_dict])
        for category in categories:
            ofile = category + file_ext
            sub_figsize = [4, 3]
            fig_col = 4
            fig_row = int((len(data_index_list) + fig_col - 1) / fig_col)
            figsize = [sub_figsize[0] * fig_col, sub_figsize[1] * fig_row]
            fig, ax = plt.subplots(fig_row, fig_col, figsize = figsize)
            target_list_of_dict = [dict_data for dict_data in spikes_list_of_dict if dict_data['category'] == category]

            sorted_target_list_of_dict = sorted(target_list_of_dict, key=lambda x:x['data_index'])
            assert all([index in [dict_data['data_index'] for dict_data in sorted_target_list_of_dict] for index in data_index_list])
            for index, data_index in enumerate(data_index_list):
                dict_data_list = [dict_tmp for dict_tmp in sorted_target_list_of_dict if dict_tmp['data_index'] == data_index]
                assert len(dict_data_list) == 1
                dict_data = dict_data_list[0]
                sound = AudioCore(dict_data['wav_file'])
                if preemphasis:
                    sound.preemphasis(coef = preemphasis_coef)
                if norm:
                    sound.normalize(target_dBFS = norm_target_dBFS)
                if align:
                    sound.align_sound(index_align = 8000, n_points = 16000, pad_zero = pad_zero)
                row, col = index // fig_col, index % fig_col
                ax_target = ax[row, col] if fig_col < len(data_index_list) else ax[col]
                sound.plot_original(ax_target)
                ax_target.set_ylim(ylim[0], ylim[1])
                ax_target.set_xlim(xlim[0], xlim[1])
                ax_target.set_xlabel('Time [s]')
                if plot_title:
                    ax_target.set_title(f'"{category}",{data_index},{os.path.basename(dict_data["wav_file"])}')
            fig.tight_layout()
            fig.savefig(ofile)
            plt.close(fig)
