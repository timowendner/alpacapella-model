import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn.functional as F
import csv
from pathlib import Path


class BeatDataset(Dataset):
    def __init__(self, csv_path: str, config: dict, split: str = "train"):
        self.config = config
        self.audio_paths = []
        self.annotations_paths = []

        csv_path = Path(csv_path)
        with open(csv_path, 'r') as f:
            metadata = list(csv.DictReader(f))

        for dataset_metadata in metadata:
            if dataset_metadata['split'] != split:
                continue
            dataset_csv_path = csv_path.parent / dataset_metadata['dataset_csv_path']
            with open(dataset_csv_path, 'r') as f:
                dataset_csv = list(csv.DictReader(f))
            for sample in dataset_csv:
                audio_path = dataset_csv_path.parent / sample["audio"]
                annotations_path = dataset_csv_path.parent / sample["annotations"]
                self.audio_paths.append(audio_path)
                self.annotations_paths.append(annotations_path)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        param = self.config['dataset']
        audio_path = self.audio_paths[idx]
        waveform, sr = torchaudio.load(str(audio_path))
        annotations_path = self.annotations_paths[idx]
        annotation = np.loadtxt(annotations_path)

        length =  int(param["input_size_seconds"] * sr)
        start = random.randint(0, max(0, waveform.shape[1] - length))
        waveform = waveform[:, start:]
        start_sec = start / sr


        data_length = param["input_size_seconds"] * sr
        waveform = waveform[:, :data_length]
        if waveform.shape[1] < data_length:
            waveform = F.pad(waveform, (0, data_length - waveform.shape[1]))


        if sr != param["sample_rate"]:
            resampler = T.Resample(sr, param["sample_rate"])
            waveform = resampler(waveform)
            sr = param["sample_rate"]

            
        spectrograms = self.compute_spectrogram(waveform, sr)
        annotation = self.compute_annotation(
            annotation, spectrograms.shape[0], start_sec, sr
        )
        return spectrograms, annotation
    
    def compute_annotation(self, annotation, total_frames, start_sec, sr):
        param = self.config['dataset']
        output = torch.zeros((total_frames, 3))
        output[:, 0] = 1
        for time, measure in annotation:
            frame = int((time - start_sec) * sr / param["hop_size"])
            if not (0 <= frame < total_frames):
                continue
            output[frame, 0] = 0
            if measure == 1:
                output[frame, 2] = 1
            else:
                output[frame, 1] = 1
        return output



    def compute_spectrogram(self, waveform, sr):
        param = self.config['dataset']

        mel_specs = []
        for window_size in param["window_sizes"]:
            mel_transform = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=window_size,
                hop_length=param["hop_size"],
                n_mels=param["n_mels"],
                f_min=30,
                f_max=sr//2,
                power=2.0
            )
            mel_spec = mel_transform(waveform)
            mel_spec = T.AmplitudeToDB()(mel_spec)
            mel_spec = mel_spec.permute(2, 1, 0)
            mel_specs.append(mel_spec)
        spectrograms = torch.cat(mel_specs, dim=2)
        return spectrograms