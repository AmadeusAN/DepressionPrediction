import torch
import torch.nn as nn
import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """encoder code

        Args:
            input_size (int): _description_
            hidden_size (int): _description_
            num_layers (int): num of encoder layers
            dropout (float): _description_
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList()

        cur_input_size = self.input_size
        if (self.input_size / (math.pow(2, num_layers))) > self.hidden_size:
            for i in range(num_layers - 1):
                output_size = cur_input_size - cur_input_size // 2
                self.encoder_layers.append(nn.Linear(cur_input_size, output_size))
                self.encoder_layers.append(nn.BatchNorm1d(output_size))
                self.encoder_layers.append(nn.ReLU())
                self.encoder_layers.append(nn.Dropout(dropout))
                cur_input_size = output_size
            self.encoder_layers.append(nn.Linear(cur_input_size, hidden_size))

        else:
            diff = (self.input_size - self.hidden_size) // self.num_layers
            for i in range(num_layers - 1):
                output_size = cur_input_size - diff
                self.encoder_layers.append(nn.Linear(cur_input_size, output_size))
                self.encoder_layers.append(nn.BatchNorm1d(output_size))
                self.encoder_layers.append(nn.ReLU())
                self.encoder_layers.append(nn.Dropout(dropout))
                cur_input_size = output_size
            self.encoder_layers.append(nn.Linear(cur_input_size, hidden_size))

    def forward(self, x):
        for module in self.encoder_layers:
            x = module(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList()
        cur_input_size = self.output_size
        if (output_size / (math.pow(2, num_layers))) > hidden_size:
            for i in range(num_layers - 1):
                output_size = cur_input_size - cur_input_size // 2
                self.decoder_layers.append(nn.Dropout(dropout))
                self.decoder_layers.append(nn.ReLU())
                self.decoder_layers.append(nn.BatchNorm1d(cur_input_size))
                self.decoder_layers.append(
                    nn.Linear(output_size, cur_input_size)
                )  # 需要交换顺序
                cur_input_size = output_size
            self.decoder_layers.append(
                nn.Linear(hidden_size, cur_input_size)
            )  # 需要交换顺序
        else:
            diff = (self.output_size - self.hidden_size) // self.num_layers
            for i in range(num_layers - 1):
                output_size = cur_input_size - diff
                self.decoder_layers.append(nn.Dropout())
                self.decoder_layers.append(nn.ReLU())
                self.decoder_layers.append(nn.BatchNorm1d(cur_input_size))
                self.decoder_layers.append(nn.Linear(output_size, cur_input_size))
                cur_input_size = output_size
            self.decoder_layers.append(nn.Linear(hidden_size, cur_input_size))

    def forward(self, x):
        for module in reversed(self.decoder_layers):
            x = module(x)
        return x


class AE(nn.Module):
    def __init__(
        self,
        input_channel: int = 384,
        output_channel: int = 1024,
        hidden_channel: int = 128,
        encoder_layer: int = 3,
        decoder_layer: int = 6,
        dropout: float = 0.1,
    ):
        super(AE, self).__init__()
        self.encoder = Encoder(
            input_size=input_channel,
            hidden_size=hidden_channel,
            num_layers=encoder_layer,
            dropout=dropout,
        )
        self.decoder = Decoder(
            hidden_size=hidden_channel,
            output_size=output_channel,
            num_layers=decoder_layer,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # if torch.isnan(x).any():
        #     raise Exception("here")
        return x


if __name__ == "__main__":
    encoder = Encoder(input_size=384, hidden_size=128, num_layers=3, dropout=0.1)
    dummy_x = torch.randn(2, 384)
    decoder = Decoder(hidden_size=128, output_size=1025, num_layers=6, dropout=0.1)

    ae = AE()
    hidden_v = encoder(dummy_x)
    print(hidden_v)
    y = decoder(hidden_v)
    y2 = ae(dummy_x)
    print(y.shape)
    print(y2.shape)
