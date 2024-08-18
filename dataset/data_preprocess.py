import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
from os.path import join

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")
TRAIN_DATASET_DIR


dir_list = os.listdir(VAL_DATASET_DIR)
dir_list = sorted(dir_list, key=int)

waveform_list = []
label_list = []
text_list = []

for dir in dir_list:
    SAMPLE_DIR = join(VAL_DATASET_DIR, dir)
    with open(join(SAMPLE_DIR, "new_label.txt")) as label_file:
        label = label_file.read()
    waveform_1, _ = torchaudio.load(join(SAMPLE_DIR, "negative_out.wav"))
    waveform_2, _ = torchaudio.load(join(SAMPLE_DIR, "neutral_out.wav"))
    waveform_3, _ = torchaudio.load(join(SAMPLE_DIR, "positive_out.wav"))
    if waveform_1.shape[0] > 1:
        waveform_1 = waveform_1[0]
    if waveform_2.shape[0] > 1:
        waveform_2 = waveform_2[0]
    if waveform_3.shape[0] > 1:
        waveform_3 = waveform_3[0]

    with open(join(SAMPLE_DIR, "negative.txt")) as text_file:
        text_1 = text_file.read()
        text_list.append(text_1)

    with open(join(SAMPLE_DIR, "neutral.txt")) as text_file:
        text_2 = text_file.read()
        text_list.append(text_2)

    with open(join(SAMPLE_DIR, "positive.txt")) as text_file:
        text_3 = text_file.read()
        text_list.append(text_3)

    label_list += [label] * 3
    waveform_list += [waveform_1, waveform_2, waveform_3]
    # break
