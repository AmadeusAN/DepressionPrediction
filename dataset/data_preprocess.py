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

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")
NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")
NDARRAY_TEST_DIR = join(NDARRAY_DIR, "test")


def get_raw_waveform_text_label(
    train: bool = True,
    binary_label: bool = True,
    concat_num: int = 3,
    resample: bool = False,
    resample_rate: int = 8000,
):
    """
    读取原始音频文本数据集，并返回waveform，text 和 label 的列表
    其中 label_new >= 53. 的抑郁症样本进行重采样，抑郁症样本数量扩充到了 6 倍。

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

        if label >= 53.0 and train:
            # 抑郁症人群需要进行随机拼接以扩充样本数量，不过当然是在训练的时候。
            waveform_list += [
                torch.cat(
                    [cur_waveform_list[x], cur_waveform_list[y], cur_waveform_list[z]]
                ).numpy()
                for x, y, z in permutations(range(3), concat_num)
            ]
            text_list += [
                cur_text_list[x] + cur_text_list[y] + cur_text_list[z]
                for x, y, z in permutations(range(3), concat_num)
            ]
            # expansion = len(permutations(range(3), concat_num))
            expansion = 6
            label_list += [label if not binary_label else 1.0] * expansion
            sample_rate_list += [sample_rate_1] * expansion

        else:
            waveform_list += [torch.cat(cur_waveform_list).numpy()]
            text_list += ["".join(cur_text_list)]
            if binary_label:
                label_list += [1.0 if label >= 53.0 else 0.0]
            else:
                label_list += [label]
            sample_rate_list += [sample_rate_1]

    label_list = np.expand_dims(np.array(label_list), axis=-1)
    return waveform_list, label_list, text_list, sample_rate_list


def get_raw_waveform_text_label_with_argumentation(
    train: bool = True,
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
            train=train,
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
            apply_noise(
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


@DeprecationWarning
def apply_effects_and_filtering(waveform, sample_rate):
    """出现了难以理解的bug

    Args:
        waveform (_type_): _description_
        sample_rate (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Define effects
    effect = ",".join(
        [
            "lowpass=frequency=300:poles=1",  # apply single-pole lowpass filter
            "atempo=0.8",  # reduce the speed
            "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3",
            # Applying echo gives some dramatic feeling
        ],
    )

    effector = torchaudio.io.AudioEffector(effect=effect)
    return effector.apply(waveform, sample_rate)


def apply_RIR(waveform, sample_rate):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.unsqueeze(torch.tensor(waveform), dim=0)
    rir = waveform[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    rir = rir / torch.linalg.vector_norm(rir, ord=2)
    augmented = F.fftconvolve(waveform, rir)
    return augmented


def apply_noise(
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
    waveform_list, label_list, text_list, sample_rate_list = (
        get_raw_waveform_text_label(
            train=True, binary_label=True, resample=True, resample_rate=4000
        )
    )

    waveform = torch.unsqueeze(torch.tensor(waveform_list[0]), dim=0)
    sample_rate = sample_rate_list[0]
    Audio(data=waveform, rate=sample_rate)

    # waveform_effect = apply_noise(
    #     waveform=waveform,
    #     sample_rate=sample_rate,
    #     single_to_noise_db_rate=10,
    # )
    # Audio(data=waveform_effect, rate=sample_rate)

    waveform_effect = apply_audio_resample(8000, waveform, sample_rate)
    Audio(data=waveform_effect, rate=8000)

    # new_waveform_list, new_label_list, new_text_list = (
    #     get_raw_waveform_text_label_with_argumentation(train=True, binary_label=True)
    # )
    # print(len(new_waveform_list))
