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
        cur_output_size = 0
        if (input_size / (math.pow(2,num_layers))) > hidden_size:
            for i in range(num_layers - 1):
                output_size = cur_input_size - cur_input_size // 2
                self.encoder_layers.append(nn.Linear(cur_input_size, output_size))
                self.encoder_layers.append(nn.ReLU())
                self.encoder_layers.append(nn.Dropout(dropout))
                cur_input_size = output_size
            self.encoder_layers.append(nn.Linear(cur_input_size, hidden_size))

        else:
            diff = (self.input_size - self.hidden_size)//self.num_layers
            for i in range(num_layers - 1):
                output_size = cur_input_size - diff
                self.encoder_layers.append(nn.Linear(cur_input_size, output_size))
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
        cur_output_size = 0
        if (output_size / (math.pow(2,num_layers))) > hidden_size:
            for i in range(num_layers - 1):
                output_size = cur_input_size - cur_input_size // 2
                self.decoder_layers.append(nn.Linear(output_size,cur_input_size))
                self.decoder_layers.append(nn.ReLU())
                self.decoder_layers.append(nn.Dropout(dropout))
                cur_input_size = output_size
            self.decoder_layers.append(nn.Linear(hidden_size,cur_input_size))
        else:
            diff = (self.output_size - self.hidden_size)//self.num_layers
            for i in range(num_layers - 1):
                output_size = cur_input_size - diff
                self.decoder_layers.append(nn.Linear(output_size,cur_input_size))
                self.decoder_layers.append(nn.ReLU())
                self.decoder_layers.append(nn.Dropout())
                cur_input_size = output_size
            self.decoder_layers.append(nn.Linear(hidden_size,cur_input_size))
            
    def forward(self, x):
        for module in reversed(self.decoder_layers):
            x = module(x)
        return x
            

if __name__ == "__main__":
    encoder = Encoder(input_size=100, hidden_size=10, num_layers=3, dropout=0.1)
    dummy_x = torch.randn(1, 100)
    
    decoder = Decoder(hidden_size=10, output_size=100, num_layers=3, dropout=0.1)
    y = decoder(encoder(dummy_x))
    print(y.shape)