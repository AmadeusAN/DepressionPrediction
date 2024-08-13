import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
from os.path import join
from onnx2torch import convert
import onnx

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")
NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")


dir_list = os.listdir(TRAIN_DATASET_DIR)
dir_list = sorted(dir_list, key=int)
dir_list
waveform_list = []
label_list = []
for dir in dir_list:
    SAMPLE_DIR = join(TRAIN_DATASET_DIR, dir)
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
    label_list += [label] * 3
    waveform_list += [waveform_1, waveform_2, waveform_3]
    # break

emotion_model = onnx.load_model(
    r"/public1/cjh/workspace/DepressionPrediction/model/component/emotion_path/model/model.onnx"
)
emotion_model = convert(emotion_model)

emotion_hiddenvecotr_list = []
for waveform in waveform_list:
    waveform = torch.unsqueeze(waveform, dim=0)
    emotion_hiddenvecotr_list.append(emotion_model(waveform)[0].detach().numpy())

emotion_hiddenvector_raw_np = np.stack(emotion_hiddenvecotr_list, axis=0)
print(emotion_hiddenvector_raw_np.shape)

np.savez_compressed(
    join(NDARRAY_TRAIN_DIR, "emotion_hiddenvector_raw.npz"),
    arr_0=emotion_hiddenvector_raw_np,
)
print("done")
