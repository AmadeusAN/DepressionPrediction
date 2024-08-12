import torch
import torch.nn as nn
from component.AE_module.ae_t import Encoder, Decoder, AE
from component.emotion_path.Wav2vec import Wav2Vec
from component.text_path.SentenceModel import SentenceModel
from component.time_frequency_path.TimeFrequencyModel import TFModel
from component.GFN_module.GFN import GFN
from component.output_module.linear_output import LinearOutput
import numpy as np
import matplotlib.pyplot as plt
import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset
import torchaudio


def preliminary_experiment():
    # load data
    audio_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.wav"
    text_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.txt"
    with open(text_path, "r") as f:
        text = f.read()
    tokens = [text]

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform[0]
    waveform = torch.unsqueeze(waveform, dim=0)
    waveform.shape

    # load model
    wav2vec = Wav2Vec()
    sentence_model = SentenceModel()
    time_frequency_model = TFModel()
    ae = AE()
    gfn = GFN()

    emotion_vector = wav2vec(waveform)[0]
    emotion_vector = torch.concat(
        [emotion_vector, torch.zeros(size=(emotion_vector.shape[0], 1))], dim=-1
    )

    text_vector = sentence_model.encode(tokens)
    text_vector_enhance = ae(torch.tensor(text_vector))
    tf_vector = time_frequency_model(waveform)
    final_vector = gfn(text_vector_enhance, emotion_vector, tf_vector)

    print(text)
    print(text_vector.shape)
    print(f"text_vector_enhance: {text_vector_enhance.shape}")
    print(emotion_vector.shape)
    print(tf_vector.shape)
    print(f"final_vector: {final_vector.shape}")


class Model(nn.Module):
    def __init__(self, output_layers: nn.Module = None):
        super(Model, self).__init__()
        self.wav2vec = Wav2Vec()
        self.sentence_model = SentenceModel()
        self.time_frequency_model = TFModel()
        self.ae = AE()
        self.gfn = GFN()
        self.output = output_layers

    def forward(self, waveform: torch.Tensor = None, text: str = None):
        """对音频向量进行前向运算

        Args:
            x (torch.Tensor): input tensor
        """
        emotion_vector = self.wav2vec(waveform)[0]
        emotion_vector = torch.concat(
            [emotion_vector, torch.zeros(size=(emotion_vector.shape[0], 1))], dim=-1
        )
        text_vector = self.sentence_model.encode(text)
        text_vector_enhance = self.ae(torch.tensor(text_vector))
        tf_vector = self.time_frequency_model(waveform)
        final_vector = self.gfn(text_vector_enhance, emotion_vector, tf_vector)

        if self.output is not None:
            final_vector = self.output(final_vector)

        return final_vector


if __name__ == "__main__":
    # load data
    audio_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.wav"
    text_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.txt"
    with open(text_path, "r") as f:
        text = f.read()
    tokens = [text]

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform[0]
    waveform = torch.unsqueeze(waveform, dim=0)

    output_layer = LinearOutput()
    model = Model(output_layers=output_layer)
    model.train()
    final_vector = model(waveform, tokens)
    print(f"final_vector: {final_vector.shape}")

    # 梯度计算实验
