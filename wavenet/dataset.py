import os

import torch
import torchaudio

from .utils import one_hot_encode


class RawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, dirpath, segment_length=5000, overlap_length=0):
        self.dirpath = dirpath
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self.files = self.get_files()
        self.length = self.get_length()

    def get_files(self):
        paths = []
        for root, folders, files in os.walk(self.dirpath):
            for file in files:
                if file.lower().endswith(('.wav', '.flac')):
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
        if isinstance(index, int):
            file_idx, segment_idx = self._file_map[index]
            file = self.files[file_idx]
            x, fs = torchaudio.load(file)
            x = x[0, :]
            hop_length = self.segment_length - self.overlap_length
            sample_start = segment_idx*hop_length
            sample_end = sample_start + self.segment_length
            segment = x[sample_start:sample_end]
            return segment
        if isinstance(index, slice):
            indexes = range(len(self))[index]
            segments = []
            for i in indexes:
                segment = self[i]
                segments.append(segment)
            segments = torch.stack(segments)
            return segments
        else:
            class_name = type(self).__name__
            index_type = type(index).__name__
            message = f'{class_name} does not support {index_type} indexing'
            raise ValueError(message)

    def __len__(self):
        return self.length


class WaveNetDataset(RawAudioDataset):
    def __init__(self, dirpath, receptive_field, target_length=32,
                 quantization_levels=256):
        self.dirpath = dirpath
        self.receptive_field = receptive_field
        self.target_length = target_length
        self.quantization_levels = quantization_levels

        segment_length = receptive_field + target_length
        self._raw_dataset = RawAudioDataset(dirpath, segment_length)

    def __repr__(self):
        kwargs = [
            'dirpath',
            'receptive_field',
            'target_length',
            'quantization_levels',
        ]
        kwargs = [f'{kwarg}={getattr(self, kwarg)}' for kwarg in kwargs]
        kwargs = ', '.join(kwargs)
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        return f'{module_name}.{class_name}({kwargs})'

    def __getitem__(self, index):
        if isinstance(index, int):
            segment = self._raw_dataset[index]
            one_hot = one_hot_encode(segment, self.quantization_levels)
            input_, target = one_hot[:, :-1], one_hot[:, -self.target_length:]
            target = target.argmax(dim=0)
            return input_, target
        if isinstance(index, slice):
            indexes = range(len(self))[index]
            inputs, targets = [], []
            for i in indexes:
                input_, target = self[i]
                inputs.append(input_)
                targets.append(target)
            inputs = torch.stack(inputs)
            targets = torch.stack(targets)
            return inputs, targets
        else:
            class_name = type(self).__name__
            index_type = type(index).__name__
            message = f'{class_name} does not support {index_type} indexing'
            raise ValueError(message)

    def __len__(self):
        return self._raw_dataset.length
