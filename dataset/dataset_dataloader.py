"""
主要的数据集加载模块
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os.path import join

sys.path.append(r"/public1/cjh/workspace/DepressionPrediction")
from dataset.data_preprocess import (
    get_raw_waveform_text_label_with_argumentation,
    get_raw_waveform_text_label,
    generate_train_val_dataset,
)
import torchaudio
import os
from sklearn.model_selection import train_test_split

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")

NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")


def get_trimodal_dataloader(batch_size: int = 1, resmaple_rate: int = 8000):
    train_dataset, val_dataset = generate_train_val_dataset(resample_rate=resmaple_rate)
    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    ), DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def get_waveform_ndarary(
    train: bool = True,
    bi_label: bool = False,
    resample: bool = True,
    resample_rate: int = 8000,
    concat_num: int = 3,
):
    """返回装有 waveform ndarray 的list，和 label 的 list

    Args:
        train (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
    # TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
    # VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")
    # if train:
    #     dir_list = os.listdir(TRAIN_DATASET_DIR)
    # else:
    #     dir_list = os.listdir(VAL_DATASET_DIR)

    # dir_list = sorted(dir_list, key=int)
    # waveform_list = []
    # label_list = (
    #     (
    #         np.load(
    #             "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/train/label_expand.npz"
    #         )["arr_0"]
    #         / 100
    #         if train
    #         else np.load(
    #             "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/test/labels.npz"
    #         )["arr_0"]
    #         / 100
    #     )
    #     if not bi_label
    #     else (
    #         np.load(
    #             "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/train/label_expand_bi.npz"
    #         )["arr_0"]
    #         if train
    #         else np.load(
    #             "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/test/label_expand_bi.npz"
    #         )["arr_0"]
    #     )
    # )

    # for dir in dir_list:
    #     SAMPLE_DIR = (
    #         join(TRAIN_DATASET_DIR, dir) if train else join(VAL_DATASET_DIR, dir)
    #     )
    #     waveform_1_raw, _ = torchaudio.load(join(SAMPLE_DIR, "negative.wav"))
    #     waveform_2_raw, _ = torchaudio.load(join(SAMPLE_DIR, "neutral.wav"))
    #     waveform_3_raw, _ = torchaudio.load(join(SAMPLE_DIR, "positive.wav"))
    #     waveform_1, _ = torchaudio.load(join(SAMPLE_DIR, "negative_out.wav"))
    #     waveform_2, _ = torchaudio.load(join(SAMPLE_DIR, "neutral_out.wav"))
    #     waveform_3, _ = torchaudio.load(join(SAMPLE_DIR, "positive_out.wav"))
    #     if waveform_1_raw.shape[0] > 1:
    #         waveform_1_raw = waveform_1_raw[0]
    #     if waveform_2_raw.shape[0] > 1:
    #         waveform_2_raw = waveform_2_raw[0]
    #     if waveform_3_raw.shape[0] > 1:
    #         waveform_3_raw = waveform_3_raw[0]

    #     if waveform_1.shape[0] > 1:
    #         waveform_1 = waveform_1[0]
    #     if waveform_2.shape[0] > 1:
    #         waveform_2 = waveform_2[0]
    #     if waveform_3.shape[0] > 1:
    #         waveform_3 = waveform_3[0]

    #     # label_list += [label] * 3
    #     waveform_list += [
    #         waveform_1_raw.numpy(),
    #         waveform_1.numpy(),
    #         waveform_2_raw.numpy(),
    #         waveform_2.numpy(),
    #         waveform_3_raw.numpy(),
    #         waveform_3.numpy(),
    #     ]
    # # break

    if train:
        waveform_list, label_list, _ = get_raw_waveform_text_label_with_argumentation(
            binary_label=bi_label,
            resample=resample,
            resample_rate=resample_rate,
            concat_num=concat_num,
        )
    else:
        waveform_list, label_list, _, _ = get_raw_waveform_text_label(
            train=False,
            binary_label=bi_label,
            resample=resample,
            resample_rate=resample_rate,
        )

    # get train_datset and test_dataset
    if train:
        X_train, X_test, y_train, y_test = train_test_split(
            waveform_list, label_list, test_size=0.2, random_state=42
        )
        return X_train, y_train, X_test, y_test

    else:
        return waveform_list, label_list


def waveform_sample():
    """仅生成一个 waveform 音频数据，供单个网络进行测试用"""
    audio_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/1/positive_out.wav"
    wavefrom, _ = torchaudio.load(audio_path)
    wavefrom = wavefrom[0]
    return torch.unsqueeze(wavefrom, dim=0)


def get_text_ndarray(
    train=True,
    bi_label: bool = False,
    resample: bool = True,
    resample_rate: int = 8000,
    concat_num: int = 3,
):
    """给定 text 文本和标签数据集，以ndarray格式的 List

    Args:
        train (bool, optional): _description_. Defaults to True.
    """
    if train:
        _, label_list, text_list = get_raw_waveform_text_label_with_argumentation(
            binary_label=bi_label,
            resample=resample,
            resample_rate=resample_rate,
            concat_num=concat_num,
        )
    else:
        _, label_list, text_list, _ = get_raw_waveform_text_label(
            train=train,
            binary_label=bi_label,
            resample=resample,
            resample_rate=resample_rate,
        )

    if train:
        X_train, X_test, y_train, y_test = train_test_split(
            text_list,
            label_list,
            test_size=0.2,
            random_state=42,
        )
        return X_train, y_train, X_test, y_test

    else:
        return text_list, label_list


def get_raw_trimodal_ndarray_dataset(
    train: bool = True,
    binary_label: bool = True,
    resample: bool = True,
    resample_rate: int = 8000,
    concat_num: int = 3,
):
    if train:
        waveform_list, label_list, text_list = (
            get_raw_waveform_text_label_with_argumentation(
                binary_label=binary_label,
                resample=resample,
                resample_rate=resample_rate,
                concat_num=concat_num,
            )
        )
    else:
        waveform_list, label_list, text_list, _ = get_raw_waveform_text_label(
            train=False,
            binary_label=binary_label,
            concat_num=concat_num,
            resample=resample,
            resample_rate=resample_rate,
        )

    # get train_datset and test_dataset
    if train:
        return train_test_split(
            waveform_list, text_list, label_list, test_size=0.2, random_state=42
        )
        # return (
        #     waveform_list_train,
        #     waveform_list_test,
        #     text_list_train,
        #     text_list_test,
        #     label_train,
        #     label_test,
        # )

    else:
        return waveform_list, text_list, label_list


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_trimodal_dataloader(batch_size=1)
    pass
