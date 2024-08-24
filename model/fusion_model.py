import torch
import torch.nn as nn

import sys

sys.path.append(r"/public1/cjh/workspace/DepressionPrediction")

from model.component.AE_module.ae_t import Encoder, Decoder, AE
from model.component.emotion_path.Wav2vec import Wav2Vec
from model.component.text_path.SentenceModel import SentenceModel
from model.component.time_frequency_path.TimeFrequencyModel import TFModel
from model.component.GFN_module.GFN import GFN
from model.component.output_module.linear_output import LinearOutput
from model.component.CSENet.dc_crn import DCCRN
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
import sys

# 子级模块中导出同级别子集模块的暂时性解决方法
sys.path.append("/public1/cjh/workspace/DepressionPrediction")
from dataset.dataset_dataloader import get_tri_modal_dataloader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # device
# 一些路径。
NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")


class Model(nn.Module):
    def __init__(self, output_layers: nn.Module = None):
        super(Model, self).__init__()
        # self.wav2vec = Wav2Vec()
        # self.sentence_model = SentenceModel()
        self.time_frequency_model = TFModel()
        self.ae = AE()
        self.gfn = GFN()
        self.output = output_layers
        self.init_param()

    def init_param(
        self,
    ):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print(f"模型参数初始化成功")

    def forward(
        self,
        waveform_tf_vec: torch.Tensor = None,
        text_vec: torch.Tensor = None,
        emotion_vec: torch.Tensor = None,
    ):
        """对音频向量进行前向运算

        Args:
            x (torch.Tensor): input tensor
        """
        # emotion_vector = self.wav2vec(waveform)[0]
        # emotion_vector = torch.concat(
        # [emotion_vector, torch.zeros(size=(emotion_vector.shape[0], 1)).to(device)],
        # dim=-1,
        # )
        # text_vector = self.sentence_model.encode(text)
        text_vector_enhance = self.ae(text_vec)
        tf_vector = self.time_frequency_model(waveform_tf_vec)
        final_vector = self.gfn(text_vector_enhance, emotion_vec, tf_vector)

        if self.output is not None:
            final_vector = self.output(final_vector)

        return final_vector


class SimpleFusionModel(nn.Module):
    def __init__(self, device: str = "cpu"):
        super(SimpleFusionModel, self).__init__()
        self.device = device
        self.sentencetransformer = SentenceModel()
        self.Waw2Vec = Wav2Vec()
        self.CSENet = DCCRN(
            rnn_units=256,
            use_clstm=True,
            kernel_num=[32, 64, 128, 256, 256, 256],
            return_hidden=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 1024 + 384, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def load_param(self):
        pass

    def forward(self, waveform, text_str):
        with torch.no_grad():
            emotion_vec = self.Waw2Vec(waveform)[0]
        text_vec = torch.unsqueeze(
            torch.tensor(self.sentencetransformer.encode(text_str)), dim=0
        ).to(self.device)
        tf_vec = self.CSENet(waveform)

        final_vec = torch.concat([emotion_vec, tf_vec, text_vec], dim=-1)
        output = self.classifier(final_vec)
        return output


if __name__ == "__main__":
    # # load data
    # audio_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.wav"
    # text_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.txt"
    # with open(text_path, "r") as f:
    #     text = f.read()
    # tokens = [text]

    # waveform, sample_rate = torchaudio.load(audio_path)
    # if waveform.shape[0] > 1:
    #     waveform = waveform[0]
    # waveform = torch.unsqueeze(waveform, dim=0).to(device)
    # dummy_y = torch.randn(size=(1, 1)).to(device)

    # output_layer = LinearOutput()
    # model = Model(output_layers=output_layer)
    # model.to(device)
    # print(f"using device: {device}")
    # model.train()

    # # 梯度计算实验
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    # loss_fn = torch.nn.MSELoss()

    # for _ in range(500):
    #     final_vector = model(waveform, tokens)
    #     # print(f"final_vector: {final_vector.shape}")
    #     loss = loss_fn(final_vector, dummy_y)
    #     print(f"loss: {loss}")
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    # train_dataloader, test_dataloader = get_tri_modal_dataloader()

    # e, t, w, l = next(iter(train_dataloader))
    # l = torch.unsqueeze(torch.tensor([float(i) for i in l]), dim=-1).to(device)
    # print(e.shape)
    # print(t.shape)
    # print(w.shape)
    # print(l.shape)

    # # 梯度计算实验
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001)
    # loss_fn = torch.nn.MSELoss()

    # for _ in range(500):
    #     final_vector = model(w.to(device), t.to(device), e.to(device))
    #     # print(f"final_vector: {final_vector.shape}")
    #     loss = loss_fn(final_vector, l)
    #     print(f"loss: {loss}")
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    model = SimpleFusionModel(device="cuda:1").to("cuda:1")
    waveform = torch.rand(1, 160000).to("cuda:1")
    text = "hello world"
    y = torch.randn(1, 1).to("cuda:1")

    # 梯度计算实验
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00000001)
    loss_fn = torch.nn.MSELoss()

    for _ in range(500):
        y_hat = model(waveform, text)
        loss = loss_fn(y_hat, y)
        print(f"loss: {loss}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
