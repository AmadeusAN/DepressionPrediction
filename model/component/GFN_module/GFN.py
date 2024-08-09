import torch
import torch.nn as nn
import torch.nn.functional as F

class modal_attention_network(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class FirstLayer(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.modal_attention_network = modal_attention_network(input_dim, 1, dropout)

    def forward(self, x_1, x_2, x_3):
        a_1 = self.modal_attention_network(x_1)
        a_2 = self.modal_attention_network(x_2)
        a_3 = self.modal_attention_network(x_3)
        U = 1 / 3 * torch.sum(torch.stack([a_1 * x_1, a_2 * x_2, a_3 * x_3], dim=1), dim=1)
        return U, a_1, a_2, a_3
    

class MultiLayerNeuralFusionNetwork(nn.Module):
    '''
    :input: 两种模态向量
    :output: 双模态节点
    '''

    def __init__(self, input_dim):
        super().__init__()
        self.fusion_layer = nn.Linear(2 * input_dim, input_dim)

    def forward(self, x_1, x_2):
        return self.fusion_layer(torch.cat([x_1, x_2], dim=1))
    

class SecondLayer(nn.Module):
    def __init__(self, k: int = 120):
        super().__init__()
        self.multilayer_neural_fusion_network = MultiLayerNeuralFusionNetwork(k)

    def forward(self, x_1, x_2, x_3, a_1, a_2, a_3):
        V_12 = self.multilayer_neural_fusion_network(x_1, x_2)
        V_13 = self.multilayer_neural_fusion_network(x_1, x_3)
        V_23 = self.multilayer_neural_fusion_network(x_2, x_3)

        S_12 = x_1 @ x_2.T
        S_13 = x_1 @ x_3.T
        S_23 = x_2 @ x_3.T

        a_12_hat = (a_1 + a_2) / (S_12 + 0.5)
        a_13_hat = (a_1 + a_3) / (S_13 + 0.5)
        a_23_hat = (a_2 + a_3) / (S_23 + 0.5)

        a_12 = torch.exp(a_12_hat) / (torch.exp(a_13_hat) + torch.exp(a_23_hat))
        a_13 = torch.exp(a_13_hat) / (torch.exp(a_12_hat) + torch.exp(a_23_hat))
        a_23 = torch.exp(a_23_hat) / (torch.exp(a_12_hat) + torch.exp(a_13_hat))

        B = torch.sum(torch.stack([a_12 * V_12, a_13 * V_13, a_23 * V_23], dim=1), dim=1)
        return B, a_12, a_13, a_23, V_12, V_13, V_23
    
    
class fusion_layer_for_thirdmodal(nn.Module):
    def __init__(self, k: int = 120):
        super().__init__()
        self.multilayer_neural_fusion_network = MultiLayerNeuralFusionNetwork(k)

    def forward(self, V_1, V_23, V_2, V_13, V_3, V_12, a_1, a_23, a_2, a_13, a_3, a_12):
        V_1_23 = self.multilayer_neural_fusion_network(V_1, V_23)
        V_2_13 = self.multilayer_neural_fusion_network(V_2, V_13)
        V_3_12 = self.multilayer_neural_fusion_network(V_3, V_12)

        S_1_23 = V_1 @ V_23.T
        S_2_13 = V_2 @ V_13.T
        S_3_12 = V_3 @ V_12.T

        a_1_23_hat = (a_1 + a_23) / (S_1_23 + 0.5)
        a_2_13_hat = (a_2 + a_13) / (S_2_13 + 0.5)
        a_3_12_hat = (a_3 + a_12) / (S_3_12 + 0.5)
        a_1_23 = torch.exp(a_1_23_hat) / (torch.exp(a_2_13_hat) + torch.exp(a_3_12_hat))
        a_2_13 = torch.exp(a_2_13_hat) / (torch.exp(a_1_23_hat) + torch.exp(a_3_12_hat))
        a_3_12 = torch.exp(a_3_12_hat) / (torch.exp(a_1_23_hat) + torch.exp(a_2_13_hat))

        return V_1_23, V_2_13, V_3_12, a_1_23, a_2_13, a_3_12

class ThirdLayer(nn.Module):
    def __init__(self, k: int = 120):
        super().__init__()
        self.fusion_module_1 = SecondLayer(k)
        self.fusion_module_2 = fusion_layer_for_thirdmodal(k)

    def forward(self, V_1, V_2, V_3, V_12, V_13, V_23, a_1, a_2, a_3, a_12, a_13, a_23):
        _, a_1213, a_1223, a_1323, V_1213, V_1223, V_1323 = self.fusion_module_1(V_12, V_13, V_23, a_12, a_13, a_23)
        V_1_23, V_2_13, V_3_12, a_1_23, a_2_13, a_3_12 = self.fusion_module_2(V_1, V_23, V_2, V_13, V_3, V_12, a_1,
                                                                              a_23, a_2, a_13, a_3, a_12)
        O = torch.sum(torch.stack(
            [a_1_23 * V_1_23, a_2_13 * V_2_13, a_3_12 * V_3_12, a_1213 * V_1213, a_1223 * V_1223, a_1323 * V_1323],
            dim=1), dim=1)
        return O
    
    
class GFN(nn.Module):
    def __init__(self, k: int = 1025, drop_out:float = 0.5):
        super().__init__()
        self.first_layer = FirstLayer(k, drop_out)
        self.second_layer = SecondLayer(k)
        self.third_layer = ThirdLayer(k)
        
    def forward(self, x_1, x_2, x_3):
        U, a_1, a_2, a_3 = self.first_layer(x_1, x_2, x_3)
        B, a_12, a_13, a_23, V_12, V_13, V_23 = self.second_layer(x_1, x_2, x_3, a_1, a_2,a_3)
        O = self.third_layer(x_1, x_2, x_3, V_12, V_13, V_23, a_1, a_2, a_3, a_12, a_13, a_23)
        return torch.concat([U, B, O], dim=1)
    

if __name__ == '__main__':
    V_1 = torch.randn(1, 1025)
    V_2 = torch.randn(1, 1025)
    V_3 = torch.randn(1, 1025)
    
    GFN = GFN(k = 1025)
    Z = GFN(V_1, V_2, V_3)
    print(Z.shape)