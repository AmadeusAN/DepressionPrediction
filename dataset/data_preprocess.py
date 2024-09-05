import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from torchaudio.utils import download_asset
import numpy as np
import os
from os.path import join
from itertools import permutations
from IPython.display import Audio
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s|%(message)s",
# )

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")
NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")
NDARRAY_TEST_DIR = join(NDARRAY_DIR, "test")
WAVFORM_NAME_LIST = ["negative_out.wav", "neutral_out.wav", "positive_out.wav"]
TEXT_NAME_LIST = ["negative.txt", "neutral.txt", "positive.txt"]


class trimodal_dataset(Dataset):
    def __init__(
        self,
        waveform_list: list,
        text_list: list,
        label_list: list,
        bianry_label: bool = True,
    ):
        super(trimodal_dataset, self).__init__()
        self.waveform_list = waveform_list
        self.text_list = text_list
        self.label_list = (
            label_list
            if bianry_label
            else torch.unsqueeze(torch.tensor(label_list, dtype=torch.float32), dim=-1)
            / 100
        )

    def __len__(self):
        return len(self.waveform_list)

    def __getitem__(self, index):
        return self.waveform_list[index], self.text_list[index], self.label_list[index]


def generate_train_val_dataset(resample_rate: int = 8000, binary_label: bool = False):
    dir_list = sorted(os.listdir(TRAIN_DATASET_DIR), key=int)
    depression_waveform_list = []
    depression_text_list = []
    depression_label_list = []
    undepression_waveform_list = []
    undepression_text_list = []
    undepression_label_list = []

    if resample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=resample_rate, lowpass_filter_width=6
        )
    else:
        resampler = lambda x: x

    # 按类别分别读取到不同列表中
    for dir in dir_list:
        SAMPLE_DIR = join(TRAIN_DATASET_DIR, dir)
        with open(join(SAMPLE_DIR, "new_label.txt")) as label_file:
            label = float(label_file.read())
        waveform_list = [
            resampler(
                torch.unsqueeze(torchaudio.load(join(SAMPLE_DIR, name))[0][0], dim=0)
            )
            for name in WAVFORM_NAME_LIST
        ]
        text_list = [open(join(SAMPLE_DIR, name)).read() for name in TEXT_NAME_LIST]

        if label >= 53.0:
            # 这些操作取消了原本的三个回答拼接在一起的操作。
            # depression_waveform_list.append(torch.concat(waveform_list, dim=-1))
            depression_waveform_list += waveform_list
            # depression_text_list.append("".join(text_list))
            depression_text_list += text_list

            depression_label_list += [label] * 3
        else:
            # undepression_waveform_list.append(torch.concat(waveform_list, dim=-1))
            # undepression_text_list.append("".join(text_list))
            undepression_waveform_list += waveform_list
            undepression_text_list += text_list

            undepression_label_list += [label] * 3

    # 先进行划分
    (
        train_depression_waveform_list,
        val_depression_waveform_list,
        train_depression_text_list,
        val_depression_text_list,
        train_depression_label_list,
        val_depression_label_list,
    ) = train_test_split(
        depression_waveform_list,
        depression_text_list,
        depression_label_list,
        test_size=0.2,
        random_state=42,
    )
    (
        train_undepression_waveform_list,
        val_undepression_waveform_list,
        train_undepression_text_list,
        val_undepression_text_list,
        train_undepression_label_list,
        val_undepression_label_list,
    ) = train_test_split(
        undepression_waveform_list,
        undepression_text_list,
        undepression_label_list,
        test_size=0.2,
        random_state=42,
    )

    # 再进行过采样
    positive_sample_num = len(train_undepression_waveform_list)
    (
        train_depression_waveform_list,
        train_depression_text_list,
        train_depression_label_list,
    ) = apply_oversample(
        train_depression_waveform_list,
        train_depression_text_list,
        train_depression_label_list,
        binary_label=binary_label,
        target_num=positive_sample_num,
    )

    # 最后进行增强
    train_waveform_list = (
        train_depression_waveform_list + train_undepression_waveform_list
    )
    train_text_list = train_depression_text_list + train_undepression_text_list
    if binary_label:
        train_label_list = torch.cat(
            [
                torch.ones(len(train_depression_waveform_list)),
                torch.zeros(len(train_undepression_waveform_list)),
            ],
            dim=-1,
        )
    else:
        train_label_list = train_depression_label_list + train_undepression_label_list

    train_waveform_list, train_text_list, train_label_list = apply_augmentation(
        train_waveform_list,
        train_text_list,
        train_label_list,
        sample_rate=resample_rate,
    )

    val_waveform_list = val_depression_waveform_list + val_undepression_waveform_list
    val_text_list = val_depression_text_list + val_undepression_text_list
    if binary_label:
        val_label_list = torch.cat(
            [
                torch.ones(len(val_depression_waveform_list)),
                torch.zeros(len(val_undepression_waveform_list)),
            ],
            dim=-1,
        )
    else:
        val_label_list = val_depression_label_list + val_undepression_label_list

    return trimodal_dataset(
        train_waveform_list,
        train_text_list,
        train_label_list,
        bianry_label=binary_label,
    ), trimodal_dataset(
        val_waveform_list, val_text_list, val_label_list, bianry_label=binary_label
    )


def get_raw_waveform_text_label(
    train: bool = True,
    binary_label: bool = True,
    # concat_num: int = 3,
    resample: bool = False,
    resample_rate: int = 8000,
):
    """
    读取原始音频文本数据集，并返回waveform，text 和 label 的列表

    Args:
        train (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    dir_list = os.listdir(TRAIN_DATASET_DIR) if train else os.listdir(VAL_DATASET_DIR)
    dir_list = sorted(dir_list, key=int)

    resampler = torchaudio.transforms.Resample(
        orig_freq=16000, new_freq=resample_rate, lowpass_filter_width=6
    )

    waveform_list = []
    sample_rate_list = []
    label_list = []
    text_list = []

    for dir in dir_list:
        SAMPLE_DIR = join(TRAIN_DATASET_DIR if train else VAL_DATASET_DIR, dir)
        with open(join(SAMPLE_DIR, "new_label.txt")) as label_file:
            label = float(label_file.read())
        waveform_1, sample_rate_1 = torchaudio.load(
            join(SAMPLE_DIR, "negative_out.wav")
        )
        waveform_2, _ = torchaudio.load(join(SAMPLE_DIR, "neutral_out.wav"))
        waveform_3, _ = torchaudio.load(join(SAMPLE_DIR, "positive_out.wav"))

        waveform_1 = waveform_1[0]
        waveform_2 = waveform_2[0]
        waveform_3 = waveform_3[0]

        if resample:
            waveform_1 = resampler(waveform_1)
            waveform_2 = resampler(waveform_2)
            waveform_3 = resampler(waveform_3)
            sample_rate_1 = resample_rate

        with open(join(SAMPLE_DIR, "negative.txt")) as text_file:
            text_1 = text_file.read()

        with open(join(SAMPLE_DIR, "neutral.txt")) as text_file:
            text_2 = text_file.read()

        with open(join(SAMPLE_DIR, "positive.txt")) as text_file:
            text_3 = text_file.read()
        cur_waveform_list = [waveform_1, waveform_2, waveform_3]
        cur_text_list = [text_1, text_2, text_3]

        # if label >= 53.0 and train:
        #     # 抑郁症人群需要进行随机拼接以扩充样本数量，不过当然是在训练的时候。
        #     waveform_list += [
        #         torch.cat(
        #             [cur_waveform_list[x], cur_waveform_list[y], cur_waveform_list[z]]
        #         ).numpy()
        #         for x, y, z in permutations(range(3), concat_num)
        #     ]
        #     text_list += [
        #         cur_text_list[x] + cur_text_list[y] + cur_text_list[z]
        #         for x, y, z in permutations(range(3), concat_num)
        #     ]
        #     # expansion = len(permutations(range(3), concat_num))
        #     expansion = 6
        #     label_list += [label if not binary_label else 1.0] * expansion
        #     sample_rate_list += [sample_rate_1] * expansion

        # else:
        waveform_list += [torch.cat(cur_waveform_list).numpy()]
        text_list += ["".join(cur_text_list)]
        if binary_label:
            label_list += [1.0 if label >= 53.0 else 0.0]
        else:
            label_list += [label]
        sample_rate_list += [sample_rate_1]

    label_list = np.expand_dims(np.array(label_list), axis=-1)
    return waveform_list, label_list, text_list, sample_rate_list


def apply_augmentation(waveform_list, text_list, label_list, sample_rate: int = 8000):
    """对传入的 waveform_list 中的音频数据进行增强
    并返回增强后的 waveform list 列表，以及配套的 text_list, label_list 列表。

    Args:
        waveform_list (_type_): _description_
        sample_rate_list (_type_): _description_
    """
    noise, _ = torchaudio.load(
        r"/public1/cjh/workspace/DepressionPrediction/dataset/sample/white_noise.wav"
    )
    new_waveform_list = []
    new_text_list = []
    new_label_list = []

    for i, (w, t, l) in enumerate(zip(waveform_list, text_list, label_list)):
        # 应用增强效果
        new_waveform_list.append(apply_RIR(w, sample_rate=sample_rate))
        new_waveform_list.append(
            apply_backgroundnoise(
                w,
                sample_rate=sample_rate,
                single_to_noise_db_rate=10,
                noise=noise,
            )
        )
        new_waveform_list.append(w)
        for _ in range(3):
            new_text_list.append(t)
            new_label_list.append(l)
    # 返回增强后的数据集
    return new_waveform_list, new_text_list, new_label_list


def apply_oversample(
    waveform_list,
    text_list,
    label_list,
    binary_label: bool = True,
    target_num: int = 1000,
):
    """将waveform_list进行过采样，以扩充数据集。

    Args:
        waveform_list (List): _description_
        target_num (int, optional): 目标数目. Defaults to 1000.

    Returns:
        _type_: _description_
    """

    new_waveform_list = []
    new_text_list = []
    new_label_list = []
    for _ in range(target_num):
        i = random.randint(0, len(waveform_list) - 1)
        new_waveform_list.append(
            # 在过采样的同时施加高斯扰动
            apply_randomgaussion_noise(waveform_list[i])
        )
        new_text_list.append(text_list[i])
        new_label_list.append(label_list[i])

    if binary_label:
        return new_waveform_list, new_text_list
    else:
        return new_waveform_list, new_text_list, new_label_list

    # return (
    #     new_waveform_list,
    #     new_text_list,
    #     new_label_list if not binary_label else new_waveform_list,
    #     new_text_list,
    # )


@DeprecationWarning
def get_raw_waveform_text_label_with_argumentation(
    binary_label: bool = True,
    resample: bool = True,
    resample_rate: int = 8000,
    concat_num: int = 3,
):
    """获得增强后的数据集

    Args:
        train (bool, optional): _description_. Defaults to True.
        binary_label (bool, optional): _description_. Defaults to True.
        resample (bool, optional): _description_. Defaults to True.
        resample_rate (int, optional): _description_. Defaults to 8000.
        concat_num (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_

    Description:
        1. 随机增强数据集，增强包括：rir、noise、原数据集。
    """

    # 首先获取原始数据集
    waveform_list, label_list, text_list, sample_rate_list = (
        get_raw_waveform_text_label(
            train=True,
            binary_label=binary_label,
            resample=resample,
            resample_rate=resample_rate,
            concat_num=concat_num,
        )
    )
    noise, _ = torchaudio.load(
        r"/public1/cjh/workspace/DepressionPrediction/dataset/sample/white_noise.wav"
    )
    new_waveform_list = []
    new_label_list = []
    new_text_list = []

    for i, (w, t, l) in enumerate(zip(waveform_list, text_list, label_list)):
        # 应用增强效果
        new_waveform_list.append(apply_RIR(w, sample_rate=sample_rate_list[i]))
        new_waveform_list.append(
            apply_backgroundnoise(
                w,
                sample_rate=sample_rate_list[i],
                single_to_noise_db_rate=10,
                noise=noise,
            )
        )
        new_waveform_list.append(w)
        for _ in range(3):
            new_text_list.append(t)
            new_label_list.append(l)
    # 返回增强后的数据集
    return new_waveform_list, new_label_list, new_text_list


def apply_RIR(waveform, sample_rate):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.unsqueeze(torch.tensor(waveform), dim=0)
    rir = waveform[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    rir = rir / torch.linalg.vector_norm(rir, ord=2)
    augmented = F.fftconvolve(waveform, rir)
    return augmented


def apply_backgroundnoise(
    waveform,
    sample_rate,
    single_to_noise_db_rate: int = 10,
    noise: torch.tensor = None,
):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.unsqueeze(torch.tensor(waveform), dim=0)

    if noise is None:
        noise, _ = torchaudio.load(
            r"/public1/cjh/workspace/DepressionPrediction/dataset/sample/white_noise.wav"
        )
    noise = noise[: waveform.shape[0], : waveform.shape[1]]

    # 对 noise 放缩到 [-1,1]
    min = torch.min(noise)
    max = torch.max(noise)
    div = max - min
    noise = (noise - min) / div

    snr_dbs = torch.tensor([single_to_noise_db_rate])
    noisy_speeches = F.add_noise(waveform, noise, snr_dbs)
    return noisy_speeches


def apply_randomgaussion_noise(waveform):
    """对单个 waveform 施加随机扰动

    Args:
        waveform (_type_): _description_
        sample_rate (_type_): _description_
    """
    return waveform + (torch.randn(waveform.shape) * waveform.std() + waveform.mean())


def apply_audio_resample(
    target_sample_rate, waveform, sample_rate, return_ndarray: bool = True
):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.unsqueeze(torch.tensor(waveform), dim=0)
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=target_sample_rate, lowpass_filter_width=6
    )
    # resample_waveform = F.resample(
    #     waveform, sample_rate, target_sample_rate, lowpass_filter_width=6
    # )
    resample_waveform = resampler(waveform)

    return resample_waveform.numpy() if return_ndarray else resample_waveform


if __name__ == "__main__":
    # waveform_list, label_list, text_list, sample_rate_list = (
    #     get_raw_waveform_text_label(
    #         train=True, binary_label=True, resample=True, resample_rate=4000
    #     )
    # )

    # waveform = torch.unsqueeze(torch.tensor(waveform_list[0]), dim=0)
    # sample_rate = sample_rate_list[0]
    # Audio(data=waveform, rate=sample_rate)

    # waveform_effect = apply_noise(
    #     waveform=waveform,
    #     sample_rate=sample_rate,
    #     single_to_noise_db_rate=10,
    # )
    # Audio(data=waveform_effect, rate=sample_rate)

    # waveform_effect = apply_audio_resample(8000, waveform, sample_rate)
    # Audio(data=waveform_effect, rate=8000)

    # new_waveform_list, new_label_list, new_text_list = (
    #     get_raw_waveform_text_label_with_argumentation(train=True, binary_label=True)
    # )
    # print(len(new_waveform_list))

    # generate_train_val_dataset()

    a, b = generate_train_val_dataset(resample_rate=8000, binary_label=False)
    pass
