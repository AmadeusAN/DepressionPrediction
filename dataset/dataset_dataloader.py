import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Data
from os.path import join

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")

NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")


class WaveformTFDataset(Dataset):
    def __init__(
        self,
    ):
        super(WaveformTFDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "waveform_tf_raw.npz"))["arr_0"]

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index]


class EmotinoHiddVecDataset(Dataset):
    def __init__(
        self,
    ):
        super(EmotinoHiddVecDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "emotion_hiddenvec_raw.npz"))[
            "arr_0"
        ]

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index]


class TextVecDataset(Dataset):
    def __init__(
        self,
    ):
        super(TextVecDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "text_raw.npz"))["arr_0"]

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index]


def get_tri_train_val_dataloader(batch_size: int = 32):
    waveform_tf_dataset = WaveformTFDataset()
    emotion_hidd_vec_dataset = EmotinoHiddVecDataset()
    text_vec_dataset = TextVecDataset()

    waveform_tf_dataset_train, waveform_tf_dataset_test = torch.utils.data.random_split(
        waveform_tf_dataset, [0.8, 0.2], torch.Generator.manual_seed(42)
    )

    emotion_hidd_vec_dataset_train, emotion_hidd_vec_dataset_test = (
        torch.utils.data.random_split(
            emotion_hidd_vec_dataset, [0.8, 0.2], torch.Generator.manual_seed(42)
        )
    )

    text_vec_dataset_train, text_vec_dataset_test = torch.utils.data.random_split(
        text_vec_dataset, [0.8, 0.2], torch.Generator.manual_seed(42)
    )

    waveform_tf_train_dataloader = DataLoader(
        waveform_tf_dataset_train, batch_size=batch_size, shuffle=True
    )
    waveform_tf_test_dataloader = DataLoader(
        waveform_tf_dataset_test, batch_size=batch_size, shuffle=True
    )

    emotion_hidd_vec_train_dataloader = DataLoader(
        emotion_hidd_vec_dataset_train, batch_size=batch_size, shuffle=True
    )
    emotion_hidd_vec_test_dataloader = DataLoader(
        emotion_hidd_vec_dataset_test, batch_size=batch_size, shuffle=True
    )
    text_vec_train_dataloader = DataLoader(
        text_vec_dataset_train, batch_size=batch_size, shuffle=True
    )
    text_vec_test_dataloader = DataLoader(
        text_vec_dataset_test, batch_size=batch_size, shuffle=True
    )

    return (
        waveform_tf_train_dataloader,
        waveform_tf_test_dataloader,
        emotion_hidd_vec_train_dataloader,
        emotion_hidd_vec_test_dataloader,
        text_vec_train_dataloader,
        text_vec_test_dataloader,
    )
