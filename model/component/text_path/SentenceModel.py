from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch


def SentenceModel():
    return SentenceTransformer(
        "/public1/cjh/workspace/DepressionPrediction/model/pretrained_model/all-MiniLM-L12-v1",
        device="cuda:1",
    )


class SentenceTransformerModel(nn.Module):
    def __init__(self, device: str = "cuda:1"):
        super(SentenceTransformerModel, self).__init__()
        self.device = device
        self.model = SentenceModel()
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: str = None):
        x = torch.tensor(self.model.encode(x)).to(self.device)
        return self.classifier(x)


if __name__ == "__main__":
    model = SentenceTransformerModel()
    y_hat = model("hello world")
    pass
