import torch
import torch.nn as nn
import math


class LinearOutput(nn.Module):
    def __init__(
        self,
        input_size: int = 3072,
        layer: int = 10,
        output_size: int = 1,
        dropout: float = 0.1,
    ):
        """简易的线性输出层，输出 SDS 分数。

        Args:
            input_size (int, optional): 输入的融入向量. Defaults to 3075.
            output_size (int, optional): 实值. Defaults to 1.
        """
        super().__init__()
        self.input_size = input_size
        self.layer = layer
        self.output_size = output_size
        self.output_layer = nn.ModuleList()

        cur_input_size = self.input_size
        if (self.input_size / (math.pow(2, self.layer))) > self.output_size:
            for i in range(self.layer - 1):
                output_size = cur_input_size - cur_input_size // 2
                self.output_layer.append(nn.Linear(cur_input_size, output_size))
                self.output_layer.append(nn.BatchNorm1d(output_size))
                self.output_layer.append(nn.ReLU())
                self.output_layer.append(nn.Dropout(dropout))
                cur_input_size = output_size
            self.output_layer.append(nn.Linear(cur_input_size, self.output_size))

        else:
            diff = (self.input_size - self.hidden_size) // self.layer
            for i in range(self.layer - 1):
                output_size = cur_input_size - diff
                self.output_layer.append(nn.Linear(cur_input_size, output_size))
                self.output_layer.append(nn.BatchNorm1d(output_size))
                self.output_layer.append(nn.ReLU())
                self.output_layer.append(nn.Dropout(dropout))
                cur_input_size = output_size
            self.output_layer.append(nn.Linear(cur_input_size, self.output_size))

    def forward(self, x):
        for layer in self.output_layer:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = LinearOutput()
    dummy_x = torch.randn(1, 3075)
    print(model(dummy_x))
    print(f"shape : {model(dummy_x).shape}")
