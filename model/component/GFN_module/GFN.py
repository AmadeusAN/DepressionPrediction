import itertools
import numpy as np
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

    # def forward(self, x_1, x_2, x_3):
    #     a_1 = self.modal_attention_network(x_1)
    #     a_2 = self.modal_attention_network(x_2)
    #     a_3 = self.modal_attention_network(x_3)
    #     U = (a_1 * x_1 + a_2 * x_2 + a_3 * x_3) / 3
    #     # (
    #     #     1
    #     #     / 3
    #     #     * torch.sum(torch.stack([a_1 * x_1, a_2 * x_2, a_3 * x_3], dim=1), dim=1)
    #     # )
    #     return U, a_1, a_2, a_3

    def forward(self, xs):
        a = [self.modal_attention_network(x) for x in xs]
        U = sum(ai * x for ai, x in zip(a, xs)) / 3
        return U, a


class MultiLayerNeuralFusionNetwork(nn.Module):
    """
    :input: 两种模态向量
    :output: 双模态节点
    """

    def __init__(self, input_dim):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim), nn.BatchNorm1d(num_features=input_dim)
        )

    def forward(self, x_1, x_2):
        return self.fusion_layer(torch.cat([x_1, x_2], dim=1))


class SecondLayer(nn.Module):
    def __init__(self, k: int = 120):
        super().__init__()
        self.multilayer_neural_fusion_network = MultiLayerNeuralFusionNetwork(k)

    def forward(self, xs, alphas):
        n = len(xs)
        V = [
            self.multilayer_neural_fusion_network(xs[a], xs[b])
            for a, b in itertools.combinations(range(n), 2)
        ]
        S = [
            torch.sum(xs[a] * xs[b], dim=1, keepdim=True)
            for a, b in itertools.combinations(range(n), 2)
        ]
        a_hat = [
            (alphas[a] + alphas[b]) / (S[i] + 0.5)
            for i, (a, b) in enumerate(itertools.combinations(range(n), 2))
        ]
        se = sum(torch.exp(a) for a in a_hat)
        a = [torch.exp(a_hat[i]) / se for i in range(n)]
        B = sum([a[i] * V[i] for i in range(n)])
        return B, a, V

    def forward0(self, x_1, x_2, x_3, a_1, a_2, a_3):
        V_12 = self.multilayer_neural_fusion_network(x_1, x_2)
        V_13 = self.multilayer_neural_fusion_network(x_1, x_3)
        V_23 = self.multilayer_neural_fusion_network(x_2, x_3)

        # S_12 = x_1 @ x_2.T

        # torch.sum(x_1 * x_2, dim=1)
        S_12 = torch.squeeze(
            torch.matmul(torch.unsqueeze(x_1, dim=1), torch.unsqueeze(x_2, dim=2)),
            dim=1,
        )
        # S_13 = x_1 @ x_3.T
        S_13 = torch.squeeze(
            torch.matmul(torch.unsqueeze(x_1, dim=1), torch.unsqueeze(x_3, dim=2)),
            dim=1,
        )
        # S_23 = x_2 @ x_3.T
        S_23 = torch.squeeze(
            torch.matmul(torch.unsqueeze(x_2, dim=1), torch.unsqueeze(x_3, dim=2)),
            dim=1,
        )

        a_12_hat = (a_1 + a_2) / (S_12 + 0.5)
        a_13_hat = (a_1 + a_3) / (S_13 + 0.5)
        a_23_hat = (a_2 + a_3) / (S_23 + 0.5)

        aaa = torch.exp(a_12_hat) + torch.exp(a_13_hat) + torch.exp(a_23_hat)
        a_12 = torch.exp(a_12_hat) / aaa
        a_13 = torch.exp(a_13_hat) / aaa
        a_23 = torch.exp(a_23_hat) / aaa

        # a_12 = a_12_hat / (a_13_hat + a_23_hat)
        # a_13 = a_13_hat / (a_12_hat + a_23_hat)
        # a_23 = a_23_hat / (a_12_hat + a_13_hat)

        B = torch.sum(
            torch.stack([a_12 * V_12, a_13 * V_13, a_23 * V_23], dim=1), dim=1
        )
        return B, a_12, a_13, a_23, V_12, V_13, V_23


class fusion_layer_for_thirdmodal(nn.Module):
    def __init__(self, k: int = 120):
        super().__init__()
        self.multilayer_neural_fusion_network = MultiLayerNeuralFusionNetwork(k)

    def forward(self, V_1, V_23, V_2, V_13, V_3, V_12, a_1, a_23, a_2, a_13, a_3, a_12):
        V_1_23 = self.multilayer_neural_fusion_network(V_1, V_23)
        V_2_13 = self.multilayer_neural_fusion_network(V_2, V_13)
        V_3_12 = self.multilayer_neural_fusion_network(V_3, V_12)

        # S_1_23 = V_1 @ V_23.T
        S_1_23 = torch.squeeze(
            torch.matmul(torch.unsqueeze(V_1, dim=1), torch.unsqueeze(V_23, dim=2)),
            dim=1,
        )
        # S_2_13 = V_2 @ V_13.T
        S_2_13 = torch.squeeze(
            torch.matmul(torch.unsqueeze(V_2, dim=1), torch.unsqueeze(V_13, dim=2)),
            dim=1,
        )
        # S_3_12 = V_3 @ V_12.T
        S_3_12 = torch.squeeze(
            torch.matmul(torch.unsqueeze(V_3, dim=1), torch.unsqueeze(V_12, dim=2)),
            dim=1,
        )

        a_1_23_hat = (a_1 + a_23) / (S_1_23 + 0.5)
        a_2_13_hat = (a_2 + a_13) / (S_2_13 + 0.5)
        a_3_12_hat = (a_3 + a_12) / (S_3_12 + 0.5)
        (torch.exp(a_2_13_hat) + torch.exp(a_3_12_hat))
        a_1_23 = torch.exp(a_1_23_hat) / (torch.exp(a_2_13_hat) + torch.exp(a_3_12_hat))
        a_2_13 = torch.exp(a_2_13_hat) / (torch.exp(a_1_23_hat) + torch.exp(a_3_12_hat))
        a_3_12 = torch.exp(a_3_12_hat) / (torch.exp(a_1_23_hat) + torch.exp(a_2_13_hat))
        # a_1_23 = a_1_23_hat / (a_2_13_hat + a_3_12_hat)
        # a_2_13 = a_2_13_hat / (a_1_23_hat + a_3_12_hat)
        # a_3_12 = a_3_12_hat / (a_1_23_hat + a_2_13_hat)

        return V_1_23, V_2_13, V_3_12, a_1_23, a_2_13, a_3_12


class ThirdLayer(nn.Module):
    def __init__(self, k: int = 120):
        super().__init__()
        self.fusion_module_1 = SecondLayer(k)
        self.fusion_module_2 = fusion_layer_for_thirdmodal(k)

    def forward(self, V_1, V_2, V_3, V_12, V_13, V_23, a_1, a_2, a_3, a_12, a_13, a_23):
        _, a_1213, a_1223, a_1323, V_1213, V_1223, V_1323 = self.fusion_module_1(
            V_12, V_13, V_23, a_12, a_13, a_23
        )
        V_1_23, V_2_13, V_3_12, a_1_23, a_2_13, a_3_12 = self.fusion_module_2(
            V_1, V_23, V_2, V_13, V_3, V_12, a_1, a_23, a_2, a_13, a_3, a_12
        )
        O = torch.sum(
            torch.stack(
                [
                    a_1_23 * V_1_23,
                    a_2_13 * V_2_13,
                    a_3_12 * V_3_12,
                    a_1213 * V_1213,
                    a_1223 * V_1223,
                    a_1323 * V_1323,
                ],
                dim=1,
            ),
            dim=1,
        )
        return O


class GFN(nn.Module):
    def __init__(self, k: int = 1024, drop_out: float = 0.5):
        super().__init__()
        self.first_layer = FirstLayer(k, drop_out)
        self.second_layer = SecondLayer(k)
        # self.third_layer = ThirdLayer(k)
        self.third_layer = SecondLayer(k)

    def forward(self, *xs):
        U, a1 = self.first_layer(xs)

        # B, a_12, a_13, a_23, V_12, V_13, V_23 = self.second_layer(
        #     x_1, x_2, x_3, a_1, a_2, a_3
        # )
        B, a2, V = self.second_layer(xs, a1)

        # O = self.third_layer(
        #     x_1, x_2, x_3, V_12, V_13, V_23, a_1, a_2, a_3, a_12, a_13, a_23
        # )
        O, _, _ = self.third_layer(list(xs) + list(V), list(a1) + list(a2))

        return torch.concat([U, B, O], dim=1)

    # def forward(self, x_1, x_2, x_3):
    #     U, a_1, a_2, a_3 = self.first_layer(x_1, x_2, x_3)

    #     # B, a_12, a_13, a_23, V_12, V_13, V_23 = self.second_layer(
    #     #     x_1, x_2, x_3, a_1, a_2, a_3
    #     # )
    #     B, a, V = self.second_layer(
    #         [x_1, x_2, x_3], [a_1, a_2, a_3]
    #     )

    #     # O = self.third_layer(
    #     #     x_1, x_2, x_3, V_12, V_13, V_23, a_1, a_2, a_3, a_12, a_13, a_23
    #     # )
    #     O, _, _ = self.third_layer(
    #         [x_1, x_2, x_3] + V, [a_1, a_2, a_3] + a
    #     )

    #     return torch.concat([U, B, O], dim=1)


if __name__ == "__main__":
    V_1 = torch.randn(2, 1024)
    V_2 = torch.randn(2, 1024)
    V_3 = torch.randn(2, 1024)

    GFN = GFN(k=1024)
    Z = GFN(V_1, V_2, V_3)
    print(Z.shape)
