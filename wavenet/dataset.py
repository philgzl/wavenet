import os

import torch
import torchaudio

from .utils import one_hot_encode


class RawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, dirpath, segment_length, overlap_length=0):
        self.dirpath = dirpath
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self.files = self.get_files()
        self.length = self.get_length()

    def get_files(self):
        paths = []
        for root, folders, files in os.walk(self.dirpath):
            for file in files:
                if file.lower().endswith('.wav'):
                    path = os.path.join(root, file)
                    paths.append(path)
        return paths

    def get_length(self):
        length = 0
        self._file_map = []
        for file_idx, file in enumerate(self.files):
            metadata = torchaudio.info(file)
            samples = metadata.num_frames
            hop_length = self.segment_length - self.overlap_length
            segments = (samples - self.segment_length)//hop_length + 1
            length += segments
            for segment_idx in range(segments):
                self._file_map.append((file_idx, segment_idx))
        return length

    def __getitem__(self, index):
        if not isinstance(index, int):
            class_name = type(self).__name__
            index_type = type(index).__name__
            message = f'{class_name} does not support {index_type} indexing'
            raise ValueError(message)
        file_idx, segment_idx = self._file_map[index]
        file = self.files[file_idx]
        x, fs = torchaudio.load(file)
        x = x[0, :]
        hop_length = self.segment_length - self.overlap_length
        sample_start = segment_idx*hop_length
        sample_end = sample_start + self.segment_length
        segment = x[sample_start:sample_end]
        return segment

    def __len__(self):
        return self.length


class WaveNetDataset(RawAudioDataset):
    def __init__(self, dirpath, output_length, receptive_field,
                 quantization_levels=256):
        segment_length = output_length + receptive_field
        super().__init__(dirpath, segment_length)
        self.output_length = output_length
        self.receptive_field = receptive_field
        self.quantization_levels = quantization_levels

    def __getitem__(self, index):
        x = super().__getitem__(index)
        x = one_hot_encode(x, self.quantization_levels)
        x, y = x[:, :-1], x[:, -self.output_length:]
        y = y.argmax(dim=0)
        return x, y
