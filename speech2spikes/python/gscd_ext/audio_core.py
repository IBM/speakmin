# (C) Copyright IBM Corp. 2024
#
# Audio core
#
import matplotlib.pyplot as plt
import librosa
import math
from pydub import AudioSegment
from scipy import interpolate
import numpy as np
from pydub.utils import (audioop, ratio_to_db, get_array_type)
import array

class AudioCore(object):
    def __init__(self, file, format = 'wav', ch_index = 0):
        assert ch_index in [0, 1], f'ch_index({ch_index}) not supported'

        self.wav_file = file
        self.sound = AudioSegment.from_file(file, format=format)
        self.channels = self.sound.channels
        self.frame_rate = self.sound.frame_rate
        self.duration = self.sound.duration_seconds # as seconds
        self.sample_width = self.sound.sample_width
        self.dBFS = self.sound.dBFS
        self.dt = 1.0 / self.frame_rate

        # get as [time, sig]
        self._create_sig()
        self._create_time()
        self._do_fft()

    def _create_sig(self):
        assert self.channels in [1, 2], f'self.channels({self.channels}) not supported'
        if self.channels == 1: # Mono
            self.sig = np.array(self.sound.get_array_of_samples(), dtype='float')
        elif self.channels == 2: # Stereo
            self.sig = np.array(self.sound.get_array_of_samples(), dtype='float')[ch_index::2]

    def _create_time(self):
        self.n_points = len(self.sig)
        self.time_s = 0.0
        self.time_e = self.duration
        self.time = np.linspace(self.time_s, self.time_e, self.n_points, endpoint = False)

    def _do_fft(self):
        self.dft_x = np.fft.fft(self.sig)
        self.dft_f = np.fft.fftfreq(self.n_points, self.dt)

    def dump_as_pwl(self, ofile, scale = 0.0894e-3, shift = 1.0, echo = True):
        string_list = []
        sig_convert = self.sig * scale + shift
        if echo:
            print(f'scale = {scale}, shift = {shift}')
            print(f'min(self.sig), max(self.sig) = {min(self.sig)}, {max(self.sig)}')
            print(f'min(sig_convert), max(sig_convert) = {min(sig_convert)}, {max(sig_convert)}')
        for time, sig in zip(self.time, sig_convert):
            string_list.append(f'{round(time, 9)} {round(sig, 6)}')
        strings = '\n'.join(string_list)

        with open(ofile, 'w') as f:
            f.write(strings)

    def echo_params(self):
        print(f'self.channels = {self.channels}')
        print(f'self.frame_rate = {self.frame_rate}')
        print(f'self.duration = {self.duration}')
        print(f'self.sample_width = {self.sample_width}')
        print(f'self.dBFS = {self.dBFS}')
        print(f'self.n_points = {self.n_points}')
        print(f'max(self.sig) = {max(self.sig)}')
        print(f'min(self.sig) = {min(self.sig)}')

    #
    # Main functions
    #-------------------------------
    def speech2spikes(self, n_mels, vth, alpha = 1.0, time_unit = 1.0, norm = 'slaney', vcsv_file_list = None,
                      leak_enable = False, leak_tau = 16000e-6):
        spikes = []
        for ch in range(n_mels):
            sig_filtered = self.bpf_melfreq(n_mels, ch, norm = norm, vcsv_file_list = vcsv_file_list)
            sig_time = self.time
            potential_t, potential_v, time_spike = self.integrate_and_fire(sig_filtered, vth, alpha=alpha,
                                                                           leak_enable=leak_enable, leak_tau=leak_tau)
            time_previous = None
            for time in time_spike:
                time_per_unit = time / time_unit
                if time_unit != 1.0:
                    time_per_unit = int(round(time_per_unit))
                time = time_per_unit
                if time != time_previous:
                    spikes.append([time, ch])
                    time_previous = time

        spikes_sorted = sorted(spikes, key = lambda x: x[0])
        return spikes_sorted

    #
    # Custom made
    #-------------------------------
    def hz2mel(self, hz, style='oshaughnessy'):
        # O'Shaughnessy 1987
        # Slaney's MATLAB Auditory Toolbox (notably not using the "1000 mels at 1000 Hz")
        assert style in ['oshaughnessy', 'slaney']

        if style == 'oshaughnessy':
            f0, m0 = 700, 2595
            mel = m0 * math.log10(hz / f0 + 1)
        elif style == 'slaney': # refer to https://en.wikipedia.org/wiki/Mel_scale
            if hz < 1000:
                mel = 3 * f / 200
            else:
                mel = 15 + 27 * math.log(hz / 1000, 6.4)
        else: # not used, but leave this as it is
            f0 = 700
            m0 = 1000 / math.log(1000 / f0 + 1) # 1442.695
            mel = m0 * math.log(hz / f0 + 1)

        return mel

    def mel2hz(self, mel, style='oshaughnessy'):
        # O'Shaughnessy 1987
        # Slaney's MATLAB Auditory Toolbox (notably not using the "1000 mels at 1000 Hz")
        assert style in ['oshaughnessy', 'slaney']

        if style == 'oshaughnessy':
            f0, m0 = 700, 2595
            hz = f0 * (10 ** (mel / m0) - 1)
        elif style == 'slaney':
            if mel < 15:
                hz = 200 * mel / 3
            else:
                hz = 1000 * 6.4 ** ((mel - 15) / 27)
        else: # not used, but leave this as it is
            f0 = 700
            m0 = 1000 / math.log(1000 / f0 + 1) # 1442.695
            hz = f0 * (math.exp(mel / m0) - 1)

        return hz

    def mel_freq_banks(self, n_mels, freq_max, style='slaney', norm=False):
        mel_max = self.hz2mel(freq_max, style=style)
        mel_step = mel_max / (n_mels + 2 - 1)
        freq_list = [self.mel2hz(mel_step * index, style=style) for index in range(n_mels + 2)]
        y_list = np.zeros([n_mels, n_mels + 2])
        for index, freq in enumerate(freq_list):
            if index in [0, n_mels + 2 - 1]:
                continue

            y = np.zeros(n_mels + 2)
            if norm:
                y[index] = 2 / (freq_list[index + 1] - freq_list[index - 1])
            else:
                y[index] = 1
            y_list[index - 1] = y

        #print(f'freq_list = {freq_list}')
        #print(f'y_list = {y_list}')

        # plots
        fig, ax = plt.subplots(figsize = (8, 6))
        for ch in range(n_mels):
            ax.plot(freq_list, y_list[ch])
        fig.tight_layout()
        fig.savefig('custom_mel_freq_bank.png')
        plt.close(fig)

    def align_sound(self, index_align = 8000, n_points = 16000, pad_zero = False):
        point_diff = int(self.n_points - n_points)
        if point_diff > 0:
            data = self.sig[0 : self.n_points - point_diff]  # removes the ending point
            self.duration = self.duration - point_diff * (self.time[1] - self.time[0])
        elif point_diff < 0:
            if pad_zero:
                data = np.append(self.sig, np.zeros(abs(point_diff))) # padding zeros
            else:
                data = np.append(self.sig, np.array([self.sig[-1]] * abs(point_diff))) # padding with the last value
            self.duration = self.duration + abs(point_diff) * (self.time[1] - self.time[0])
        else:
            data = self.sig

        w_average_index = np.sum( np.abs(data) * np.arange(n_points) ) / np.sum( np.abs(data) + 1e-6) # weighted average (index)
        delta_index = index_align - int(w_average_index)
        if delta_index > 0:
            if pad_zero:
                converted_data = np.insert(data, 0, np.zeros(delta_index))[0 : n_points]
            else:
                converted_data = np.insert(data, 0, np.array([data[0]] * delta_index))[0 : n_points]
        else:
            delta_index = np.abs(delta_index)
            if pad_zero:
                converted_data = np.append(data, np.zeros(delta_index))[delta_index : (delta_index + n_points)]
            else:
                converted_data = np.append(data, np.array([data[-1]] * delta_index))[delta_index : (delta_index + n_points)]

        self.sig = converted_data
        self._create_time()
        self._do_fft()

    def normalize(self, target_dBFS = -31.782): # -24 - 7.782
        # 24 dBu = 0 dBFS (assumption)
        # -10 dBV = -7.782 dBu (line input)
        # -24 dBu - 7.782 dBu --> -31.782 dBFS
        current_dBFS = self.dBFS
        change_in_dBFS = target_dBFS - current_dBFS

        self.sig = 10 ** (change_in_dBFS / 20) * self.sig
        self._do_fft()

    def get_dBFS(self):
        sample_width = 2 # bytes
        bits = sample_width * 8 # bits
        sig_array = array.array(get_array_type(bits, signed=True), self.sig.astype(np.int16))
        rms = audioop.rms(sig_array, sample_width)
        max_possible_val = (2 ** bits)
        max_possible_amplitude = max_possible_val / 2
        return ratio_to_db(rms / max_possible_amplitude)
    #
    # By librosa
    #-------------------------------
    def preemphasis(self, coef=0.97):
        # Apply pre-emphasis filter to reduce noise
        self.sig = librosa.effects.preemphasis(self.sig, coef=coef)
        self.dBFS = self.get_dBFS()
        self._do_fft()

    def bpf_melfreq(self, n_mels, index, norm = 'slaney', echo = False, vcsv_file_list = None):
        assert norm in ['slaney', None]

        # Mel filter bank [n_mels, n_fft/2]
        melfb_freq = librosa.fft_frequencies(sr = self.frame_rate, n_fft = self.n_points)

        if vcsv_file_list != None:
            melfb = self.gen_melfb_from_vcsv(vcsv_file_list[index], melfb_freq)
        else:
            melfb_2d = librosa.filters.mel(sr = self.frame_rate, n_fft = self.n_points, n_mels = n_mels, norm = norm)
            melfb = melfb_2d[index]

        if self.n_points % 2 == 0:
            melfb_full = np.concatenate([melfb.copy()[0:len(melfb) - 1], melfb.copy()[::-1]])
        else:
            melfb_full = np.concatenate([melfb.copy(), melfb.copy()[::-1]])
        melfb_full = melfb_full[0 : len(melfb_full) - 1]

        if echo:
            print(f'self.n_points = {self.n_points}')
            print(f'len(melfb) = {len(melfb)}, len(melfb_full) = {len(melfb_full)}, len(self.dft_x) = {len(self.dft_x)}')
            print(f'self.dft_f[0] = {self.dft_f[0]}')
            print(f'self.dft_f[1] = {self.dft_f[1]}')
            print(f'self.dft_f[2] = {self.dft_f[2]}')
            print(f'self.dft_f[{int(len(self.dft_f)//2 - 1)}] = {self.dft_f[int(len(self.dft_f)//2 - 1)]}')
            print(f'self.dft_f[{int(len(self.dft_f)//2)}] = {self.dft_f[int(len(self.dft_f)//2)]}')
            print(f'self.dft_f[{int(len(self.dft_f)//2 + 1)}] = {self.dft_f[int(len(self.dft_f)//2 + 1)]}')
            print(f'self.dft_f[-3] = {self.dft_f[-3]}')
            print(f'self.dft_f[-2] = {self.dft_f[-2]}')
            print(f'self.dft_f[-1] = {self.dft_f[-1]}')
            print(f'melfb_freq[0] = {melfb_freq[0]}')
            print(f'melfb_freq[1] = {melfb_freq[1]}')
            print(f'melfb_freq[2] = {melfb_freq[2]}')
            print(f'melfb_freq[{int(len(melfb_freq)//2 - 1)}] = {melfb_freq[int(len(melfb_freq)//2 - 1)]}')
            print(f'melfb_freq[{int(len(melfb_freq)//2)}] = {melfb_freq[int(len(melfb_freq)//2)]}')
            print(f'melfb_freq[{int(len(melfb_freq)//2 + 1)}] = {melfb_freq[int(len(melfb_freq)//2 + 1)]}')
            print(f'melfb_freq[-3] = {melfb_freq[-3]}')
            print(f'melfb_freq[-2] = {melfb_freq[-2]}')
            print(f'melfb_freq[-1] = {melfb_freq[-1]}')

        dft_filtered = melfb_full * self.dft_x.copy()
        sig_filtered = np.real(np.fft.ifft(dft_filtered))

        #print(f'index={index}, argmax(melfb)={np.argmax(melfb)}, freq={melfb_freq[np.argmax(melfb)]}')
        return sig_filtered

    @classmethod
    def plot_librosa_melfreq_bfp(cls, ax, n_mels, sr = 16000, n_fft = 16000, norm = 'slaney'):
        melfb_2d = librosa.filters.mel(sr = sr, n_fft = n_fft, n_mels = n_mels, norm = norm)
        melfb_freq = librosa.fft_frequencies(sr = sr, n_fft = n_fft)
        for melfb in melfb_2d:
            ax.plot(melfb_freq, melfb)

    #
    # BPF from circuit designs
    #------------------------------
    @classmethod
    def load_vcsv(cls, vcsv_file):
        '''
        vcsv_file: extracted from ADE explore (VCSV file by Export Waveforms)
        '''
        with open(vcsv_file, 'r') as f:
            lines = f.read().splitlines()
            list_of_freq = []
            list_of_value = []
            for line in lines:
                if line.startswith(';'): # comment line
                    continue

                freq, val_real, val_imag = line.split(',')
                value = complex(float(val_real), float(val_imag))
                list_of_freq.append(float(freq))
                list_of_value.append(value)

        return list_of_freq, list_of_value

    def gen_melfb_from_vcsv(self, vcsv_file, np_freq_list):
        list_of_freq, list_of_value = AudioCore.load_vcsv(vcsv_file)
        x = np.array(list_of_freq)
        y = np.array(list_of_value)
        f_real = interpolate.interp1d(x, np.real(y))
        f_imag = interpolate.interp1d(x, np.imag(y))
        ynew_real = f_real(np_freq_list) # interpolation by `interp1d`
        ynew_imag = f_imag(np_freq_list) # interpolation by `interp1d`
        return ynew_real + 1j * ynew_imag

    #
    # Intentration & Fire (IAF)
    #------------------------------
    def integrate_and_fire(self, vin, vth, alpha=1, leak_enable=False, leak_tau=16000e-6):
        '''
        vin: BP-filtered data input (generated by self.bpf_melfreq())
        vth: threshold for creating spikes
        alpha: coeeficient for integrating potential
        '''
        time_spike_list = []
        potential_t = []
        potential_v = []
        potential = 0
        for index, v in enumerate(vin):
            t = self.time[index]

            #--- skip first data ----
            if index == 0:
                potential_t.append(t)
                potential_v.append(potential)
                s = 0
                t_pre = t
                v_pre = v
                continue

            #--- integrate ----------
            if v * v_pre < 0: # crossing 0 V --> triangle x 2
                t_0 = t_pre + abs(v_pre / v) * (t - t_pre)
                s = abs((t_0 - t_pre) * v_pre / 2) + abs((t - t_0) * v / 2)
            else:
                s = abs(v_pre + v) * (t - t_pre) / 2

            potential_pre = potential
            if leak_enable:
                potential = potential * math.exp(- (t - t_pre) / leak_tau) + s * alpha
            else:
                potential += s * alpha

            #--- Vth comparison -----
            if vth <= potential:
                while vth <= potential:
                    time_spike = t_pre + (t - t_pre) * (vth - potential_pre) / (potential - potential_pre)
                    potential_t.append(time_spike)
                    potential_v.append(vth)
                    potential_next = potential - vth
                    potential_pre = potential
                    potential = 0
                    potential_t.append(time_spike)
                    potential_v.append(potential)
                    potential_pre = potential
                    potential = potential_next
                    if potential < vth:
                        potential_t.append(t)
                        potential_v.append(potential)
                    time_spike_list.append(time_spike)
            else:
                potential_t.append(t)
                potential_v.append(potential)

            v_pre = v
            t_pre = t

        return potential_t, potential_v, time_spike_list

    #
    # Plot utilities
    #------------------------------
    # fig, ax = plt.subplots(figsize = (8, 6))
    def plot_mel_filter_banks(self, ax, n_mels, norm = 'slaney', echo=True):
        melfb_2d = librosa.filters.mel(sr = self.frame_rate, n_fft = self.n_points, n_mels = n_mels, norm = norm)
        melfb_freq = librosa.fft_frequencies(sr = self.frame_rate, n_fft = self.n_points)
        for index, melfb in enumerate(melfb_2d):
            ax.plot(melfb_freq, melfb)
            if echo:
                print(f'ch={index}, freq={melfb_freq[np.argmax(melfb)]}, {melfb[np.argmax(melfb)]}')

    def plot_dft(self, ax):
        ax.plot(self.dft_f, np.real(self.dft_x))

    def plot_original(self, ax):
        ax.plot(self.time, self.sig)
