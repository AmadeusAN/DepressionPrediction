import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset
import torchaudio
from os.path import join

AUDIO_PADDING_LENGTH = 300000  # 例如，填充到 1 秒的长度，假设采样率为 16 kHz
window_length = 2048  # 窗口长度
hop_length = 512  # 窗口滑动步长

# 一些路径。
NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def pad_audio(signal, target_length):
    B, L = signal.shape
    # 计算需要填充的长度
    padding_length = target_length - L
    if padding_length > 0:
        # 使用零填充
        signal = torch.cat([signal, torch.zeros(B, padding_length).to(device)], dim=-1)
    return signal


class SE_module(nn.Module):
    def __init__(self, in_channel: int = 1024, k: int = 2048):
        super(SE_module, self).__init__()
        self.in_channel = in_channel
        self.k = k
        self.sequeeze = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.extract = nn.Sequential(
            nn.Linear(self.in_channel, self.k),
            nn.ReLU(),
            nn.Linear(self.k, self.in_channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x:shape: (batch_size,C, N, T)
        """

        u_r = self.sequeeze(x[:, 0, :, :])  # b,1025,1
        u_i = self.sequeeze(x[:, 1, :, :])
        u_r = torch.squeeze(input=u_r, dim=-1)  # b,1025
        u_i = torch.squeeze(input=u_i, dim=-1)
        a_r = torch.unsqueeze(self.extract(u_r), dim=-1)  # b,1025,1
        a_i = torch.unsqueeze(self.extract(u_i), dim=-1)

        x_r_enhance = x[:, 0, :, :] * a_r
        x_i_enhance = x[:, 1, :, :] * a_i

        output = torch.stack([x_r_enhance, x_i_enhance], dim=1)
        return output


class Shrink(nn.Module):
    def __init__(self, shrink_size: int = 1024, H_size: int = 1024):
        super(Shrink, self).__init__()
        self.shrink_size = shrink_size
        self.inner_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, shrink_size // 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, shrink_size // 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(H_size, 1)),
        )

    def forward(self, x):
        x = self.inner_net(x)
        b, _, _, _ = x.shape
        x = torch.squeeze(x)
        if b > 1:
            return x
        else:
            return torch.unsqueeze(x, dim=0)


class TFModel(nn.Module):
    def __init__(self, need_padding: bool = True):
        super(TFModel, self).__init__()
        self.need_padding = need_padding
        self.se_module = SE_module()
        self.shrink = Shrink()

    def forward(self, x):
        # 数据预处理后，决定直接将 STFT 后的数据送入模型，而非在模型中进行 STFT。
        # if self.need_padding:
        #     x = pad_audio(x, AUDIO_PADDING_LENGTH)
        # x = torch.stft(
        #     x, n_fft=window_length, hop_length=hop_length, return_complex=True
        # )
        # x = torch.stack([x.real, x.imag], dim=1)
        x = self.se_module(x)
        x = self.shrink(x)
        return x


if __name__ == "__main__":
    timefrequncymodel = TFModel().to(device)
    # audio_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.wav"
    # waveform, sample_rate = torchaudio.load(audio_path)
    # if waveform.shape[0] > 1:
    # waveform = waveform[0]
    # waveform = torch.unsqueeze(waveform, dim=0).to(device)
    # y = timefrequncymodel(waveform)
    # print(y.shape)

    data_np = np.load(join(NDARRAY_TRAIN_DIR, "waveform_tf_raw.npz"))["arr_0"]

    print(data_np.shape)

    test_input = torch.unsqueeze(torch.tensor(data_np[0]), dim=0).to(device)

    y = timefrequncymodel(test_input)

    print(y.shape)
