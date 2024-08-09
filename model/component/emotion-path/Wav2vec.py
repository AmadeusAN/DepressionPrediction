import torch
import torch.nn as nn
import audonnx
import onnx2torch
from onnx2torch import convert
import onnx


def Wav2Vec():
    model = audonnx.load_model(r"/public1/cjh/workspace/DepressionPrediction/model/component/emotion-path/model/model.onnx")
    return convert(model)
