import os
import numpy as np
import torch
from dataset import dataset_dataloader

from model.model import Model
from model.component.output_module.linear_output import LinearOutput
from torch.optim.lr_scheduler import MultiStepLR
import sys

CUR = os.path.dirname(__file__)
SOR_DIR = os.path.dirname(CUR)
CHECKPOINT_DIR = os.path.join(SOR_DIR, "checkpoint")  # 模型权重保存目录
VISUALIZE_DIR = os.path.join(SOR_DIR, "visualize")  # 损失数据保存目录
VISUALIZE_TRAIN_DIR = os.path.join(VISUALIZE_DIR, "train")
VISUALIZE_TEST_DIR = os.path.join(VISUALIZE_DIR, "test")
SAVE_MODEL_NAME = "model_v1"  # 模型权重文件名称
SAVE_LOSS_NAME = ["mse_loss", "r2_score", "rmse_loss"]  # 用到的损失名称
BATCH_SIZE = 32
START_LEARNING_RATE = 0.000001
LR_MILESTONES = [30, 60, 90, 120, 180]

device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def r2_loss_fn(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def main(load_epoch: int = 0, end_epoch: int = 100, save_interval: int = 20):
    """
    主要训练逻辑

    :param save_interval: 每个{save_interval}个轮次保存一次模型权重。
    :param load_epoch: 准备继续训练时加载的模型权重的轮次。
    :param end_epoch:
    :return:
    """
    print(f"use {device}")

    train_dataloader, test_dataloader = dataset_dataloader.get_tri_modal_dataloader()

    # loading model
    output_layer = LinearOutput()
    model = Model(output_layers=output_layer)
    mse_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LEARNING_RATE)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=LR_MILESTONES, gamma=0.5)

    # loading parameters
    if load_epoch != 0:
        model.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, SAVE_MODEL_NAME + f"_{load_epoch}"))
        )
        print(f"parameters loaded")

    model.to(device)
    model.train()
    MSE_Loss = []
    R2_Score = []
    RMSE_Loss = []
    TEST_MSE_Loss = []
    TEST_R2_Score = []
    TEST_RMSE_Loss = []
    cur_epoch = 0

    try:
        # main training logic
        size = len(train_dataloader.dataset)
        for eph in range(load_epoch + 1, end_epoch + 1):
            print(f"epoch:{eph} ", "*" * 100)
            epoch_loss = 0.0
            epoch_r2 = 0.0
            epoch_rmse = 0.0
            for batch, (e, t, w, y) in enumerate(train_dataloader):
                e, t, w, y = e.to(device), t.to(device), w.to(device), y.to(device)
                y_hat = model(w, t, e)
                mse_loss = mse_loss_fn(y_hat, y)
                r2_score = r2_loss_fn(y_hat, y)
                rmse_loss = torch.sqrt(mse_loss)

                mse_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # record loss
                current, mse_loss, r2_score, rmse_loss = (
                    (batch + 1) * len(e),
                    mse_loss.item(),
                    r2_score.item(),
                    rmse_loss.item(),
                )
                epoch_loss += mse_loss
                epoch_r2 += r2_score
                epoch_rmse += rmse_loss
                print(
                    f"mse_loss = {mse_loss:>6f},rmse_loss = {rmse_loss:>6f}, r2_score = {r2_score:>6f},---[{current:>5d}/{size:>5d}]"
                )

            epoch_loss /= len(train_dataloader)
            epoch_r2 /= len(train_dataloader)
            epoch_rmse /= len(train_dataloader)
            MSE_Loss.append(epoch_loss)
            R2_Score.append(epoch_r2)
            RMSE_Loss.append(epoch_rmse)
            print(
                f"train mse_lost = {epoch_loss:>12f}, rmse_loss = {epoch_rmse:>12f}, r2 = {epoch_r2:>12f}"
            )
            scheduler.step()
            # ===============================================================================================================================
            # testing logic
            model.eval()
            num_batches = len(test_dataloader)
            (
                test_mse_loss,
                test_r2_score,
                test_rmse_loss,
            ) = (0.0, 0.0, 0.0)
            with torch.no_grad():
                for e, t, w, y in test_dataloader:
                    e, t, w, y = e.to(device), t.to(device), w.to(device), y.to(device)

                    y_hat = model(w, t, e)
                    mse_loss = mse_loss_fn(y_hat, y)
                    r2_score = r2_loss_fn(y_hat, y)
                    rmse_loss = torch.sqrt(mse_loss)

                    test_mse_loss += mse_loss.item()
                    test_r2_score += r2_score.item()
                    test_rmse_loss += rmse_loss.item()
            test_mse_loss /= num_batches
            test_r2_score /= num_batches
            test_rmse_loss /= num_batches
            TEST_MSE_Loss.append(test_mse_loss)
            TEST_R2_Score.append(test_r2_score)
            TEST_RMSE_Loss.append(test_rmse_loss)
            print(
                f"test  mse_loss = {test_mse_loss:>12f}, rmse_loss = {test_rmse_loss:>12f}, r2 = {test_r2_score:>12f}"
            )
            model.train()

            # save model parameters each [save_interval] epochs
            if eph % save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, SAVE_MODEL_NAME + f"_{str(eph)}"),
                )

            cur_epoch = eph

    finally:
        print(f"Are you sure to save loss data? [y/n]")
        y = input()
        if y == "y" or y == "Y":
            print(f"saving loss data" + "*" * 20)

            # 创建对应的损失文件夹
            if not os.path.exists(VISUALIZE_TRAIN_DIR):
                os.mkdir(VISUALIZE_TRAIN_DIR)
            if not os.path.exists(VISUALIZE_TEST_DIR):
                os.mkdir(VISUALIZE_TEST_DIR)

            np.savez(
                os.path.join(
                    VISUALIZE_TRAIN_DIR,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=MSE_Loss,
            )

            np.savez(
                os.path.join(
                    VISUALIZE_TEST_DIR,
                    f"test_" + SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=TEST_MSE_Loss,
            )

            np.savez(
                os.path.join(
                    VISUALIZE_TRAIN_DIR,
                    SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=R2_Score,
            )

            np.savez(
                os.path.join(
                    VISUALIZE_TEST_DIR,
                    f"test_" + SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=TEST_R2_Score,
            )

            np.savez(
                os.path.join(
                    VISUALIZE_TRAIN_DIR,
                    SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=RMSE_Loss,
            )

            np.savez(
                os.path.join(
                    VISUALIZE_TEST_DIR,
                    f"test_" + SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=TEST_RMSE_Loss,
            )

            print(f"save success")
        else:
            print(f"save cancel, discard data")


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"{device} detected")

    main(load_epoch=0, end_epoch=360, save_interval=20)
    # print(sys.path)
