import torch
import torch.nn as nn
from component.AE_module.ae_t import Encoder, Decoder, AE
from component.emotion_path.Wav2vec import Wav2Vec
from component.text_path.SentenceModel import SentenceModel
from component.time_frequency_path.TimeFrequencyModel import TFModel
import numpy as np
import matplotlib.pyplot as plt
import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset
import torchaudio


if __name__ == '__main__':
    # load data
    audio_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.wav"
    text_path = r"/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus/train/2/positive.txt"
    with open(text_path, 'r') as f:
        text = f.read()
    tokens = [text]
    
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform[0]
    waveform = torch.unsqueeze(waveform,dim = 0)
    waveform.shape
    
    # load model
    wav2vec = Wav2Vec()
    sentence_model = SentenceModel()
    time_frequency_model = TFModel()
    ae = AE()
    
    
    emotion_vector = wav2vec(waveform)[0]
    emotion_vector = torch.concat([emotion_vector,torch.zeros(size = (emotion_vector.shape[0],1))],dim = -1)
    
    text_vector = sentence_model.encode(tokens)
    text_vector_enhance = ae(torch.tensor(text_vector))
    tf_vector = time_frequency_model(waveform)
    print(text)
    print(text_vector.shape)
    print(text_vector_enhance.shape)
    print(emotion_vector.shape)
    print(tf_vector.shape)