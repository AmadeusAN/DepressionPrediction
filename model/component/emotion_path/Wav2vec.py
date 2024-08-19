import torch.nn as nn
from onnx2torch import convert
import torch
import onnx
import sys

sys.path.append(r"/public1/cjh/workspace/DepressionPrediction")
from dataset.dataset_dataloader import waveform_sample


def Wav2Vec():
    model = onnx.load_model(
        r"/public1/cjh/workspace/DepressionPrediction/model/component/emotion_path/model/model.onnx"
    )
    return convert(model)


class Wav2VecModel(nn.Module):
    def __init__(self):
        super(Wav2VecModel, self).__init__()
        self.wav2vec = Wav2Vec()
        self.classifer = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.wav2vec(x)
        x = self.classifer(x[0])
        return x


if __name__ == "__main__":

    wavefrom = waveform_sample()
    y = torch.rand(size=(1, 1))
    model = Wav2VecModel()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(100):
        y_hat = model(wavefrom)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        print(loss.item())

        optimizer.zero_grad()
