"""
主要的数据集加载模块
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os.path import join
from dataset.data_preprocess import get_raw_waveform_text_label_with_argumentation
import torchaudio
import os
from sklearn.model_selection import train_test_split

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")

NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")


class WaveformTFDataset(Dataset):
    def __init__(self, label_dataset: np.ndarray = None):
        super(WaveformTFDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "waveform_tf_raw.npz"))["arr_0"]
        self.label = label_dataset

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index], self.label[index]


class EmotinoHiddVecDataset(Dataset):
    def __init__(self, label_dataset: np.ndarray = None):
        super(EmotinoHiddVecDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "emotion_hiddenvec_raw.npz"))[
            "arr_0"
        ]
        self.label = label_dataset

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index], self.label[index]


class TextVecDataset(Dataset):
    def __init__(self, label_dataset: np.ndarray = None):
        super(TextVecDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "text_raw.npz"))["arr_0"]
        self.label = label_dataset

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index], self.label[index]


class TriModalDataset(Dataset):
    """元素顺序: e, t, w, l

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
    ):
        super(TriModalDataset, self).__init__()
        self.emotion_np = np.load(join(NDARRAY_TRAIN_DIR, "emotion_hiddenvec_raw.npz"))[
            "arr_0"
        ]

        self.text_np = np.load(join(NDARRAY_TRAIN_DIR, "text_raw.npz"))["arr_0"]
        self.waveform_tf_np = np.load(join(NDARRAY_TRAIN_DIR, "waveform_tf_raw.npz"))[
            "arr_0"
        ]

        self.label = np.load(join(NDARRAY_TRAIN_DIR, "labels.npz"))["arr_0"] / 100

    def __len__(
        self,
    ):
        return len(self.emotion_np)

    def __getitem__(self, index):
        return (
            self.emotion_np[index],
            self.text_np[index],
            self.waveform_tf_np[index],
            self.label[index],
        )


# def get_tri_train_val_dataloader(batch_size: int = 32):
#     label_dataset = np.load(join(NDARRAY_TRAIN_DIR, "labels.npz"))["arr_0"]
#     waveform_tf_dataset = WaveformTFDataset(label_dataset)
#     emotion_hidd_vec_dataset = EmotinoHiddVecDataset(label_dataset)
#     text_vec_dataset = TextVecDataset(label_dataset)

#     waveform_tf_dataset_train, waveform_tf_dataset_test = torch.utils.data.random_split(
#         waveform_tf_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
#     )

#     emotion_hidd_vec_dataset_train, emotion_hidd_vec_dataset_test = (
#         torch.utils.data.random_split(
#             emotion_hidd_vec_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
#         )
#     )

#     text_vec_dataset_train, text_vec_dataset_test = torch.utils.data.random_split(
#         text_vec_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
#     )

#     waveform_tf_train_dataloader = DataLoader(
#         waveform_tf_dataset_train, batch_size=batch_size, shuffle=True
#     )
#     waveform_tf_test_dataloader = DataLoader(
#         waveform_tf_dataset_test, batch_size=batch_size, shuffle=True
#     )

#     emotion_hidd_vec_train_dataloader = DataLoader(
#         emotion_hidd_vec_dataset_train, batch_size=batch_size, shuffle=True
#     )
#     emotion_hidd_vec_test_dataloader = DataLoader(
#         emotion_hidd_vec_dataset_test, batch_size=batch_size, shuffle=True
#     )
#     text_vec_train_dataloader = DataLoader(
#         text_vec_dataset_train, batch_size=batch_size, shuffle=True
#     )
#     text_vec_test_dataloader = DataLoader(
#         text_vec_dataset_test, batch_size=batch_size, shuffle=True
#     )

#     return (
#         waveform_tf_train_dataloader,
#         waveform_tf_test_dataloader,
#         emotion_hidd_vec_train_dataloader,
#         emotion_hidd_vec_test_dataloader,
#         text_vec_train_dataloader,
#         text_vec_test_dataloader,
#     )


def get_tri_modal_dataloader(batch_size: int = 32):
    """该方法获得的训练集和验证集都是已经经过特征提取后的特征向量，没有原始数据

    Args:
        batch_size (int, optional): _description_. Defaults to 32.

    Returns:
        dataloader: 训练和验证
    """
    tri_modal_dataset = TriModalDataset()
    tri_modal_dataset_train, tri_modal_dataset_test = torch.utils.data.random_split(
        tri_modal_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
    )

    tri_modal_dataloader_train = DataLoader(
        tri_modal_dataset_train, batch_size=batch_size, shuffle=True
    )
    tri_modal_dataloader_test = DataLoader(
        tri_modal_dataset_test, batch_size=batch_size, shuffle=False
    )
    return tri_modal_dataloader_train, tri_modal_dataloader_test


def get_waveform_ndarary(train: bool = True, bi_label: bool = False):
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

    waveform_list, label_list, _ = get_raw_waveform_text_label_with_argumentation(
        train=train, binary_label=bi_label
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
    if wavefrom.shape[0] > 1:
        wavefrom = wavefrom[0]
    return torch.unsqueeze(wavefrom, dim=0)


def get_text_ndarray(train=True, bi_label: bool = False):
    """给定 text 文本和标签数据集，以ndarray格式的 List

    Args:
        train (bool, optional): _description_. Defaults to True.
    """
    DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
    TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
    VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")
    text_list = []
    label_list = []
    if train:
        dir_list = os.listdir(TRAIN_DATASET_DIR)
    else:
        dir_list = os.listdir(VAL_DATASET_DIR)

    dir_list = sorted(dir_list, key=int)
    if not bi_label:
        label_list = (
            np.load(
                "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/train/labels.npz"
            )["arr_0"]
            / 100
            if train
            else np.load(
                "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/test/labels.npz"
            )["arr_0"]
            / 100
        )

    for dir in dir_list:
        SAMPLE_DIR = (
            join(TRAIN_DATASET_DIR, dir) if train else join(VAL_DATASET_DIR, dir)
        )

        with open(join(SAMPLE_DIR, "negative.txt")) as text_file:
            text_1 = text_file.read()
            text_list.append(text_1)

        with open(join(SAMPLE_DIR, "neutral.txt")) as text_file:
            text_2 = text_file.read()
            text_list.append(text_2)

        with open(join(SAMPLE_DIR, "positive.txt")) as text_file:
            text_3 = text_file.read()
            text_list.append(text_3)

        if bi_label:
            with open(join(SAMPLE_DIR, "new_label.txt")) as label_file:
                label = label_file.read()
            label_list += [label] * 3

    if bi_label:
        label_list_np = np.array(label_list, dtype=np.float32)
        label_list_bi = [1 if x >= 53 else 0 for x in label_list_np]
        label_list_bi = np.array(label_list_bi, dtype=np.float32)
        label_list_bi = np.expand_dims(label_list_bi, axis=-1)
    # get train_datset and test_dataset
    if train:
        X_train, X_test, y_train, y_test = train_test_split(
            text_list,
            label_list_bi if bi_label else label_list,
            test_size=0.2,
            random_state=42,
        )
        return X_train, y_train, X_test, y_test

    else:
        return text_list, label_list if not bi_label else label_list_bi


def get_raw_trimodal_ndarray_dataset(
    train: bool = True,
):
    DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
    TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
    VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")
    if train:
        dir_list = os.listdir(TRAIN_DATASET_DIR)
    else:
        dir_list = os.listdir(VAL_DATASET_DIR)

    dir_list = sorted(dir_list, key=int)
    label_list = (
        np.load(
            "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/train/labels.npz"
        )["arr_0"]
        / 100
        if train
        else np.load(
            "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray/test/labels.npz"
        )["arr_0"]
        / 100
    )

    waveform_list = []
    text_list = []
    for dir in dir_list:
        SAMPLE_DIR = (
            join(TRAIN_DATASET_DIR, dir) if train else join(VAL_DATASET_DIR, dir)
        )

        waveform_1, _ = torchaudio.load(join(SAMPLE_DIR, "negative_out.wav"))
        waveform_2, _ = torchaudio.load(join(SAMPLE_DIR, "neutral_out.wav"))
        waveform_3, _ = torchaudio.load(join(SAMPLE_DIR, "positive_out.wav"))
        if waveform_1.shape[0] > 1:
            waveform_1 = waveform_1[0]
        if waveform_2.shape[0] > 1:
            waveform_2 = waveform_2[0]
        if waveform_3.shape[0] > 1:
            waveform_3 = waveform_3[0]

        waveform_list += [waveform_1.numpy(), waveform_2.numpy(), waveform_3.numpy()]

        with open(join(SAMPLE_DIR, "negative.txt")) as text_file:
            text_1 = text_file.read()
            text_list.append(text_1)

        with open(join(SAMPLE_DIR, "neutral.txt")) as text_file:
            text_2 = text_file.read()
            text_list.append(text_2)

        with open(join(SAMPLE_DIR, "positive.txt")) as text_file:
            text_3 = text_file.read()
            text_list.append(text_3)

    # break

    # get train_datset and test_dataset
    if train:
        (
            waveform_list_train,
            waveform_list_test,
            text_list_train,
            text_list_test,
            label_train,
            label_test,
        ) = train_test_split(
            waveform_list, text_list, label_list, test_size=0.2, random_state=42
        )
        return (
            waveform_list_train,
            waveform_list_test,
            text_list_train,
            text_list_test,
            label_train,
            label_test,
        )

    else:
        return waveform_list, text_list, label_list


if __name__ == "__main__":
    # (
    #     waveform_tf_train_dataloader,
    #     waveform_tf_test_dataloader,
    #     emotion_hidd_vec_train_dataloader,
    #     emotion_hidd_vec_test_dataloader,
    #     text_vec_train_dataloader,
    #     text_vec_test_dataloader,
    # ) = get_tri_train_val_dataloader()

    # train_data, train_label = next(iter(waveform_tf_train_dataloader))
    # print(train_data.shape)
    # print(train_label.shape)

    # dataset = TriModalDataset()
    # e, t, w, l = dataset[0]
    # print(e.shape)
    # print(t.shape)
    # print(w.shape)
    # print(l.shape)

    # train_input, train_label = get_tri_modal_dataloader()
    # e, t, w, l = next(iter(train_input))
    # print(e.shape)
    # print(t.shape)
    # print(w.shape)
    # print(l.shape)

    # X_train, y_train, X_test, y_test = get_text_ndarray(train=True, bi_label=True)

    waveform_list_train, label_list_train, waveform_list_test, label_list_test = (
        get_waveform_ndarary(bi_label=True)
    )
    print(len(waveform_list_train))

    # waveform_list_test, label_list_test = get_waveform_ndarary(train=False)
    # print(len(waveform_list_test))

    # text_list_train, label_list_train, text_list_test, label_list_test = (
    #     get_text_ndarray()
    # )
    # print(len(text_list_train))
    # (
    #     waveform_list_train,
    #     waveform_list_test,
    #     text_list_train,
    #     text_list_test,
    #     label_train,
    #     label_test,
    # ) = get_raw_trimodal_ndarray_dataset()
    # pass
