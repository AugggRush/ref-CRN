#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sys
import numpy as np
from scipy.io.wavfile import read


class wavTrans(object):

    def __init__(self, winLen, hopLen):
        self.winLen = winLen
        self.hopLen = hopLen
        self.audioLen = -1

    def winGenerate(self):
        bins = torch.arange(0.5, self.winLen, dtype=float)
        win = torch.sin(bins / self.winLen * np.pi)
        return win

    def wav2frames(self, rawWave_tensor):
        # rawWave should be a torch tensor
        self.audioLen = len(rawWave_tensor)
        num_frame = int(torch.ceil(torch.tensor(self.audioLen / self.hopLen)).item())
        frames = torch.zeros([int(num_frame), self.winLen])
        win = self.winGenerate()
        audioPadded = torch.cat([rawWave_tensor, torch.zeros(int((num_frame-1) * self.hopLen + self.winLen - self.audioLen))])
        winAdjust = torch.zeros_like(audioPadded, dtype=float)
        for i in range(num_frame):
            winAdjust[i*self.hopLen:i*self.hopLen+self.winLen] = winAdjust[i*self.hopLen:i*self.hopLen+self.winLen] + (win * win)
        winAdjust = torch.sqrt(winAdjust*self.winLen)
        for t in range(num_frame):
            oneFrame = audioPadded[t*self.hopLen:t*self.hopLen+self.winLen] * win \
                / winAdjust[t*self.hopLen:t*self.hopLen+self.winLen]
            frames[t] = oneFrame
        return frames

    def frame2wav(self, frame_tensor):
        row, col = frame_tensor.size()
        assert (col == self.winLen)
        win = self.winGenerate()
        winAdjust = torch.zeros(int((row - 1)*self.hopLen + col), dtype=float)
        for i in range(row):
            winAdjust[i*self.hopLen:i*self.hopLen+col] = winAdjust[i*self.hopLen:i*self.hopLen+col] + (win * win)
        winAdjust = torch.sqrt(winAdjust / col)
        placeHolder = torch.zeros(int((row - 1) * self.hopLen + col), dtype = np.float)
        for iframe in range(row):
            placeHolder[iframe*self.hopLen:iframe*self.hopLen+col] = placeHolder[iframe*self.hopLen:iframe*self.hopLen+col] + \
		            frame_tensor[iframe] * win / winAdjust[iframe*self.hopLen:iframe*self.hopLen+col]
        if self.audioLen == -1:
            audio = placeHolder
        else:
            audio = placeHolder[:self.audioLen]
        return audio

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

class LJspeechDataset(Dataset):

    def __init__(self, training_files, win_length, hop_length, sampling_rate):
        self.audio_files = files_to_list(training_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.trans = wavTrans(self.win_length, self.hop_length)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # Read audio file
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        audio = audio / 32768.0
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # clip wave into frames
        frames_tensor = self.trans.wav2frames(audio)

        return frames_tensor


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_len = [s.size(0) for s in data]
    for ind, l in enumerate(data_len):
        if l <= 500:
            continue
        else:
            max_start = l - 500
            data_start = random.randint(0, max_start)
            data[ind] = data[ind][data_start:data_start+500]
    data = pad_sequence(data, batch_first=True)
    return data