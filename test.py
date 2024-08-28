"""
tensorboard 测试
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model.fusion_model import SimpleFusionModel
from model.component.CSENet.dc_crn import DCCRN

if __name__ == "__main__":
    writer = SummaryWriter("./tensorboard_logs")
    image = torch.rand(3, 256, 256)
    # writer.add_image("image", image, global_step=10)

    # mse_loss = torch.rand(size=(1, 1))
    # rmse_loss = torch.sqrt(mse_loss)

    # writer.add_scalars(
    #     main_tag="Loss",
    #     tag_scalar_dict={"MSE": mse_loss, "RMSE": rmse_loss},
    #     global_step=0,
    # )
    # writer.flush()

    # model = DCCRN(
    #     rnn_units=256,
    #     use_clstm=True,
    #     kernel_num=[32, 64, 128, 256, 256, 256],
    # )
    # dummy_w = torch.rand(size=(1, 16000))
    # dumny_y = torch.randn(size=(1, 1))

    # writer.add_graph(model, dummy_w)
    writer.add_embedding(mat=image[0])
    writer.flush()
