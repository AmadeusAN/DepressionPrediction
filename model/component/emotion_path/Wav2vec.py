import torch.nn as nn
from onnx2torch import convert
import onnx


def Wav2Vec():
    model = onnx.load_model(r"/public1/cjh/workspace/DepressionPrediction/model/component/emotion_path/model/model.onnx")
    return convert(model)

if __name__ == "__main__":
    model = Wav2Vec()
    print(model)
