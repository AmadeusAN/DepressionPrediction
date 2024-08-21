import os
import numpy as np
import torch
import torch.nn as nn
from dataset import dataset_dataloader
from dataset.dataset_dataloader import (
    get_waveform_ndarary,
    get_text_ndarray,
    get_raw_trimodal_ndarray_dataset,
)
from model.fusion_model import Model
from model.component.output_module.linear_output import LinearOutput
from torch.optim.lr_scheduler import MultiStepLR
from model.component.CSENet.dc_crn import DCCRN
from model.component.emotion_path.Wav2vec import Wav2VecModel
from model.component.text_path.SentenceModel import SentenceTransformerModel
from model.fusion_model import SimpleFusionModel
from sklearn.metrics import precision_score, recall_score, f1_score

import sys

# CUR = os.path.dirname(__file__)
SOR_DIR = os.path.dirname(__file__)
checkpint_dir = os.path.join(SOR_DIR, "checkpoint")  # 模型权重保存目录
VISUALIZE_DIR = os.path.join(SOR_DIR, "visualize")  # 损失数据保存目录
visualize_train_dir = os.path.join(VISUALIZE_DIR, "train")
visualize_val_dir = os.path.join(VISUALIZE_DIR, "val")
save_model_name = "model_v1"  # 模型权重文件名称
# SAVE_LOSS_NAME = ["mse_loss", "r2_score", "rmse_loss"]  # 用到的损失名称
SAVE_LOSS_NAME = ["bce_loss"]
BATCH_SIZE = 16
START_LEARNING_RATE = 1e-4
LR_MILESTONES = [10, 20, 30, 50, 80, 100, 200]

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

    train_dataloader, test_dataloader = dataset_dataloader.get_tri_modal_dataloader(
        batch_size=BATCH_SIZE
    )

    # loading model
    output_layer = LinearOutput()
    model = Model(output_layers=output_layer)
    mse_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LEARNING_RATE)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=LR_MILESTONES, gamma=0.5)

    # loading parameters
    if load_epoch != 0:
        model.load_state_dict(
            torch.load(os.path.join(checkpint_dir, save_model_name + f"_{load_epoch}"))
        )
        print(f"parameters loaded")

    model.to(device)
    model.train()
    MSE_Loss = []
    R2_Score = []
    RMSE_Loss = []
    VAL_MSE_Loss = []
    VAL_R2_Score = []
    VAL_RMSE_Loss = []
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
                val_mse_loss,
                val_r2_score,
                val_rmse_loss,
            ) = (0.0, 0.0, 0.0)
            with torch.no_grad():
                for e, t, w, y in test_dataloader:
                    e, t, w, y = e.to(device), t.to(device), w.to(device), y.to(device)

                    y_hat = model(w, t, e)
                    mse_loss = mse_loss_fn(y_hat, y)
                    r2_score = r2_loss_fn(y_hat, y)
                    rmse_loss = torch.sqrt(mse_loss)

                    val_mse_loss += mse_loss.item()
                    val_r2_score += r2_score.item()
                    val_rmse_loss += rmse_loss.item()
            val_mse_loss /= num_batches
            val_r2_score /= num_batches
            val_rmse_loss /= num_batches
            VAL_MSE_Loss.append(val_mse_loss)
            VAL_R2_Score.append(val_r2_score)
            VAL_RMSE_Loss.append(val_rmse_loss)
            print(
                f"val  mse_loss = {val_mse_loss:>12f}, rmse_loss = {val_rmse_loss:>12f}, r2 = {val_r2_score:>12f}"
            )
            model.train()

            # save model parameters each [save_interval] epochs
            if eph % save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpint_dir, save_model_name + f"_{str(eph)}"),
                )

            cur_epoch = eph

    finally:
        print(f"Are you sure to save loss data? [y/n]")
        y = input()
        if y == "y" or y == "Y":
            print(f"saving loss data" + "*" * 20)

            # 创建对应的损失文件夹
            if not os.path.exists(visualize_train_dir):
                os.mkdir(visualize_train_dir)
            if not os.path.exists(visualize_val_dir):
                os.mkdir(visualize_val_dir)

            np.savez(
                os.path.join(
                    visualize_train_dir,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=MSE_Loss,
            )

            np.savez(
                os.path.join(
                    visualize_val_dir,
                    f"test_" + SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_MSE_Loss,
            )

            np.savez(
                os.path.join(
                    visualize_train_dir,
                    SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=R2_Score,
            )

            np.savez(
                os.path.join(
                    visualize_val_dir,
                    f"test_" + SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_R2_Score,
            )

            np.savez(
                os.path.join(
                    visualize_train_dir,
                    SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=RMSE_Loss,
            )

            np.savez(
                os.path.join(
                    visualize_val_dir,
                    f"test_" + SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_RMSE_Loss,
            )

            print(f"save success")
        else:
            print(f"save cancel, discard data")


def train_single_modal(
    model: torch.nn.Module = None,
    text_path: bool = False,
    bi_label: bool = False,
    load_epoch: int = 0,
    end_epoch: int = 100,
    save_interval: int = 10,
    checkpint_dir: str = None,
    save_model_name: str = None,
    visualize_train_dir: str = None,
    visualize_val_dir: str = None,
    start_lr: float = 1e-4,
):
    """
    主要训练逻辑

    :param save_interval: 每个{save_interval}个轮次保存一次模型权重。
    :param load_epoch: 准备继续训练时加载的模型权重的轮次。
    :param end_epoch:
    :return:
    """
    print(f"use {device}")

    if text_path:
        data_list_train, label_list_train, data_list_val, label_list_val = (
            get_text_ndarray()
        )
    else:
        data_list_train, label_list_train, data_list_val, label_list_val = (
            get_waveform_ndarary(train=tuple, bi_label=bi_label)
        )

    # mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=LR_MILESTONES, gamma=0.1)

    # loading parameters
    if load_epoch != 0:
        model.load_state_dict(
            torch.load(os.path.join(checkpint_dir, save_model_name + f"_{load_epoch}"))
        )
        print(f"parameters loaded")

    model.to(device)
    model.train()
    # MSE_Loss = []
    # RMSE_Loss = []
    BCE_Loss = []
    # VAL_MSE_Loss = []
    # VAL_RMSE_Loss = []
    VAL_BCE_Loss = []
    cur_epoch = 0

    try:
        # main training logic
        size = len(data_list_train)
        for eph in range(load_epoch + 1, end_epoch + 1):
            print(f"epoch:{eph} ", "*" * 100)
            # epoch_mse = 0.0
            # epoch_rmse = 0.0
            epoch_bce = 0.0
            for index, (x, y) in enumerate(zip(data_list_train, label_list_train)):
                x, y = (
                    torch.unsqueeze(torch.tensor(x).to(device), dim=0)
                    if not text_path
                    else x
                ), torch.unsqueeze(
                    torch.tensor(y, dtype=torch.float32).to(device), dim=0
                )
                y_hat = model(x)

                # mse_loss = mse_loss_fn(y_hat, y)
                # rmse_loss = torch.sqrt(mse_loss)
                bce_loss = bce_loss_fn(y_hat, y)

                # mse_loss.backward()
                bce_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # record loss
                current = index + 1
                # mse_loss, rmse_loss = (
                #     mse_loss.item(),
                #     rmse_loss.item(),
                # )
                bce_loss = bce_loss.item()
                # epoch_mse += mse_loss
                # epoch_rmse += rmse_loss
                epoch_bce += bce_loss
                # print(
                #     f"mse_loss = {mse_loss:>6f},rmse_loss = {rmse_loss:>6f} ---[{current:>5d}/{size:>5d}]"
                # )
                print(f"bce_loss = {bce_loss:>12f}---[{current:>5d}/{size:>5d}]")

            # epoch_mse /= len(data_list_train)
            # epoch_rmse /= len(data_list_train)
            epoch_bce /= size
            # MSE_Loss.append(epoch_mse)
            # RMSE_Loss.append(epoch_rmse)
            BCE_Loss.append(epoch_bce)
            # print(f"train mse_lost = {epoch_mse:>12f}, rmse_loss = {epoch_rmse:>12f}")
            print(f"train bce_lost = {epoch_bce:>12f}")
            scheduler.step()
            # ===============================================================================================================================
            # validation logic
            model.eval()
            num_batches = len(data_list_val)
            # (
            #     val_mse_loss,
            #     val_rmse_loss,
            # ) = (0.0, 0.0)
            val_bce_loss = 0.0
            with torch.no_grad():
                for x, y in zip(data_list_val, label_list_val):
                    x, y = (
                        torch.unsqueeze(torch.tensor(x).to(device), dim=0)
                        if not text_path
                        else x
                    ), torch.unsqueeze(torch.tensor(y,dtype=torch.float32).to(device), dim=0)

                    y_hat = model(x)
                    # mse_loss = mse_loss_fn(y_hat, y)
                    # rmse_loss = torch.sqrt(mse_loss)
                    bce_loss = bce_loss_fn(y_hat, y)

                    # val_mse_loss += mse_loss.item()
                    # val_rmse_loss += rmse_loss.item()
                    val_bce_loss += bce_loss.item()
            # val_mse_loss /= num_batches
            # val_rmse_loss /= num_batches
            val_bce_loss /= num_batches
            # VAL_MSE_Loss.append(val_mse_loss)
            # VAL_RMSE_Loss.append(val_rmse_loss)
            VAL_BCE_Loss.append(val_bce_loss)
            # print(
            #     f"val  mse_loss = {val_mse_loss:>12f}, rmse_loss = {val_rmse_loss:>12f}"
            # )
            print(f"val  bce_loss = {val_bce_loss:>12f}")
            model.train()

            # save model parameters each [save_interval] epochs
            if eph % save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpint_dir, save_model_name + f"_{str(eph)}"),
                )

            cur_epoch = eph

    finally:
        print(f"Are you sure to save loss data? [y/n]")
        y = input()
        if y == "y" or y == "Y":
            print(f"saving loss data" + "*" * 20)

            # 创建对应的损失文件夹
            if not os.path.exists(visualize_train_dir):
                os.mkdir(visualize_train_dir)
            if not os.path.exists(visualize_val_dir):
                os.mkdir(visualize_val_dir)

            np.savez(
                os.path.join(
                    visualize_train_dir,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=BCE_Loss,
            )

            np.savez(
                os.path.join(
                    visualize_val_dir,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_BCE_Loss,
            )

            # np.savez(
            #     os.path.join(
            #         visualize_train_dir,
            #         SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
            #     ),
            #     data=MSE_Loss,
            # )

            # np.savez(
            #     os.path.join(
            #         visualize_val_dir,
            #         SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
            #     ),
            #     data=VAL_MSE_Loss,
            # )

            # np.savez(
            #     os.path.join(
            #         visualize_train_dir,
            #         SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
            #     ),
            #     data=RMSE_Loss,
            # )

            # np.savez(
            #     os.path.join(
            #         visualize_val_dir,
            #         SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
            #     ),
            #     data=VAL_RMSE_Loss,
            # )

            print(f"save success")
        else:
            print(f"save cancel, discard data")


def test_single_modal(
    model: nn.Module = None,
    text_path: bool = False,
    load_epoch: int = 0,
    checkpint_dir: str = None,
    save_model_name: str = None,
):

    print(f"use {device}")
    data_list_test, label_list_test = (
        get_text_ndarray(train=False)
        if text_path
        else get_waveform_ndarary(train=False)
    )
    dataset_len = len(data_list_test)
    mse_loss_fn = torch.nn.BCEWithLogitsLoss()
    (
        test_mse_loss,
        test_rmse_loss,
    ) = (0.0, 0.0)
    predict = []
    groundtruth = []

    model.load_state_dict(
        torch.load(os.path.join(checkpint_dir, save_model_name + f"_{load_epoch}"))
    )
    print(f"parameters loaded")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in zip(data_list_test, label_list_test):

            groundtruth.append(1 if y >= 0.53 else 0)

            x, y = (x if text_path else torch.tensor(x).to(device)), torch.unsqueeze(
                torch.tensor(y).to(device), dim=0
            )

            y_hat = torch.unsqueeze(model(x), dim=0)

            predict.append(1 if y_hat.item() >= 0.53 else 0)
            mse_loss = mse_loss_fn(y_hat, y)
            rmse_loss = torch.sqrt(mse_loss)
            test_mse_loss += mse_loss.item()
            test_rmse_loss += rmse_loss.item()
        test_mse_loss /= dataset_len
        test_rmse_loss /= dataset_len
        print(
            f"test  mse_loss = {test_mse_loss:>12f}, rmse_loss = {test_rmse_loss:>12f}"
        )

    # 进行分类计算
    y_pred = np.array(predict)
    y_true = np.array(groundtruth)

    # true positive
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    print(f"TP = {TP}")

    # false positive
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    print(f"FP = {FP}")

    # true negative
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    print(f"TN = {TN}")

    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    print(f"FN = {FN}")

    p = precision_score(y_true, y_pred, average="binary")
    r = recall_score(y_true, y_pred, average="binary")
    f1score = f1_score(y_true, y_pred, average="binary")

    print(f"precision = {p}, recall = {r}, f1score = {f1score}")


def train_fusion_model(
    model: torch.nn.Module = None,
    load_epoch: int = 0,
    end_epoch: int = 100,
    save_interval: int = 10,
    checkpint_dir: str = None,
    save_model_name: str = None,
    visualize_train_dir: str = None,
    visualize_val_dir: str = None,
    start_lr: float = 1e-4,
):
    (
        waveform_list_train,
        waveform_list_test,
        text_list_train,
        text_list_test,
        label_train,
        label_test,
    ) = get_raw_trimodal_ndarray_dataset()

    mse_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=LR_MILESTONES, gamma=0.1)

    # loading parameters
    if load_epoch != 0:
        model.load_state_dict(
            torch.load(os.path.join(checkpint_dir, save_model_name + f"_{load_epoch}"))
        )
        print(f"parameters loaded")

    model.to(device)
    model.train()
    MSE_Loss = []
    RMSE_Loss = []
    VAL_MSE_Loss = []
    VAL_RMSE_Loss = []
    cur_epoch = 0

    try:
        # main training logic
        size = len(waveform_list_train)
        for eph in range(load_epoch + 1, end_epoch + 1):
            print(f"epoch:{eph} ", "*" * 100)
            epoch_mse = 0.0
            epoch_rmse = 0.0
            for index, (w, t, y) in enumerate(
                zip(waveform_list_train, text_list_train, label_train)
            ):
                w, y = (
                    torch.unsqueeze(torch.tensor(w).to(device), dim=0)
                ), torch.unsqueeze(torch.tensor(y).to(device), dim=0)
                y_hat = model(w, t)

                mse_loss = mse_loss_fn(y_hat, y)
                rmse_loss = torch.sqrt(mse_loss)

                mse_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # record loss
                current, mse_loss, rmse_loss = (
                    index + 1,
                    mse_loss.item(),
                    rmse_loss.item(),
                )
                epoch_mse += mse_loss
                epoch_rmse += rmse_loss
                print(
                    f"mse_loss = {mse_loss:>6f},rmse_loss = {rmse_loss:>6f} ---[{current:>5d}/{size:>5d}]"
                )

            epoch_mse /= size
            epoch_rmse /= size
            MSE_Loss.append(epoch_mse)
            RMSE_Loss.append(epoch_rmse)
            print(f"train mse_lost = {epoch_mse:>12f}, rmse_loss = {epoch_rmse:>12f}")
            scheduler.step()
            # ===============================================================================================================================
            # validation logic
            model.eval()
            num_batches = len(waveform_list_test)
            (
                val_mse_loss,
                val_rmse_loss,
            ) = (0.0, 0.0)
            with torch.no_grad():
                for w, t, y in zip(waveform_list_test, text_list_test, label_test):
                    w, y = (
                        torch.unsqueeze(torch.tensor(w).to(device), dim=0)
                    ), torch.unsqueeze(torch.tensor(y).to(device), dim=0)

                    y_hat = model(w, t)
                    mse_loss = mse_loss_fn(y_hat, y)
                    rmse_loss = torch.sqrt(mse_loss)

                    val_mse_loss += mse_loss.item()
                    val_rmse_loss += rmse_loss.item()
            val_mse_loss /= num_batches
            val_rmse_loss /= num_batches
            VAL_MSE_Loss.append(val_mse_loss)
            VAL_RMSE_Loss.append(val_rmse_loss)
            print(
                f"val  mse_loss = {val_mse_loss:>12f}, rmse_loss = {val_rmse_loss:>12f}"
            )
            model.train()

            # save model parameters each [save_interval] epochs
            if eph % save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpint_dir, save_model_name + f"_{str(eph)}"),
                )

            cur_epoch = eph

    finally:
        print(f"Are you sure to save loss data? [y/n]")
        y = input()
        if y == "y" or y == "Y":
            print(f"saving loss data" + "*" * 20)

            # 创建对应的损失文件夹
            if not os.path.exists(visualize_train_dir):
                os.mkdir(visualize_train_dir)
            if not os.path.exists(visualize_val_dir):
                os.mkdir(visualize_val_dir)

            np.savez(
                os.path.join(
                    visualize_train_dir,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=MSE_Loss,
            )

            np.savez(
                os.path.join(
                    visualize_val_dir,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_MSE_Loss,
            )

            np.savez(
                os.path.join(
                    visualize_train_dir,
                    SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=RMSE_Loss,
            )

            np.savez(
                os.path.join(
                    visualize_val_dir,
                    SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_RMSE_Loss,
            )

            print(f"save success")
        else:
            print(f"save cancel, discard data")


def test_fusion_model(
    model: nn.Module = None,
    load_epoch: int = 0,
    checkpint_dir: str = None,
    save_model_name: str = None,
):
    print(f"use {device}")
    waveform_list, text_list, label_list = get_raw_trimodal_ndarray_dataset(train=False)
    dataset_len = len(waveform_list)
    mse_loss_fn = torch.nn.MSELoss()
    (
        test_mse_loss,
        test_rmse_loss,
    ) = (0.0, 0.0)
    predict = []
    groundtruth = []

    model.load_state_dict(
        torch.load(os.path.join(checkpint_dir, save_model_name + f"_{load_epoch}"))
    )
    print(f"parameters loaded")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for w, t, y in zip(waveform_list, text_list, label_list):

            groundtruth.append(1 if y >= 0.53 else 0)

            w = torch.tensor(w).to(device)
            y = torch.unsqueeze(torch.tensor(y).to(device), dim=0)

            y_hat = model(w, t)

            predict.append(1 if y_hat.item() >= 0.53 else 0)
            mse_loss = mse_loss_fn(y_hat, y)
            rmse_loss = torch.sqrt(mse_loss)
            test_mse_loss += mse_loss.item()
            test_rmse_loss += rmse_loss.item()
        test_mse_loss /= dataset_len
        test_rmse_loss /= dataset_len
        print(
            f"test  mse_loss = {test_mse_loss:>12f}, rmse_loss = {test_rmse_loss:>12f}"
        )

    # 进行分类计算
    y_pred = np.array(predict)
    y_true = np.array(groundtruth)

    # true positive
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    print(f"TP = {TP}")

    # false positive
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    print(f"FP = {FP}")

    # true negative
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    print(f"TN = {TN}")

    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    print(f"FN = {FN}")

    p = precision_score(y_true, y_pred, average="binary")
    r = recall_score(y_true, y_pred, average="binary")
    f1score = f1_score(y_true, y_pred, average="binary")

    print(f"precision = {p}, recall = {r}, f1score = {f1score}")


if __name__ == "__main__":
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available() else "cpu"
    # )

    # 启用异常检测模式
    # torch.autograd.set_detect_anomaly(True)
    # torch.use_deterministic_algorithms(True)

    print(f"{device} detected")

    # main(load_epoch=0, end_epoch=360, save_interval=20)
    # print(sys.path)

    # print(VISUALIZE_TRAIN_DIR)

    # load DCCRN
    # model = DCCRN(
    #     rnn_units=256,
    #     use_clstm=True,
    #     kernel_num=[32, 64, 128, 256, 256, 256],
    # )

    # load Wav2Vec
    model = Wav2VecModel()

    # load SentenceTransformer
    # model = SentenceTransformerModel(device=device)

    train_single_modal(
        model=model,
        text_path=False,
        bi_label=True,
        load_epoch=0,
        end_epoch=100,
        save_interval=20,
        checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/Wav2Vec_expand",
        save_model_name="Wav2Vec_expand",
        visualize_train_dir="/public1/cjh/workspace/DepressionPrediction/visualize/train/Wav2Vec_expand",
        visualize_val_dir="/public1/cjh/workspace/DepressionPrediction/visualize/val/Wav2Vec_expand",
        start_lr=1e-3,
    )

    # test_single_modal(
    #     model=model,
    #     # text_path=True,
    #     load_epoch=20,
    #     checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/CSENet",
    #     save_model_name="CSENet",
    # )

    # load fusion model
    # model = SimpleFusionModel(device=device)
    # train_fusion_model(
    #     model=model,
    #     load_epoch=0,
    #     end_epoch=100,
    #     save_interval=20,
    #     checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/fusion_model",
    #     save_model_name="fusion_model",
    #     visualize_train_dir="/public1/cjh/workspace/DepressionPrediction/visualize/train/fusion_model",
    #     visualize_val_dir="/public1/cjh/workspace/DepressionPrediction/visualize/val/fusion_model",
    #     start_lr=1e-3,
    # )

    # test_fusion_model(
    #     model=model,
    #     load_epoch=20,
    #     checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/fusion_model",
    #     save_model_name="fusion_model",
    # )
