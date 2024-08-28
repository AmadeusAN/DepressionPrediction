import os
import numpy as np
import torch
import torch.nn as nn
from dataset import dataset_dataloader
from dataset.dataset_dataloader import (
    get_waveform_ndarary,
    get_text_ndarray,
    get_raw_trimodal_ndarray_dataset,
    get_trimodal_dataloader,
)
from dataset.data_preprocess import get_raw_waveform_text_label
from model.fusion_model import Model
from model.component.output_module.linear_output import LinearOutput
from torch.optim.lr_scheduler import MultiStepLR
from model.component.CSENet.dc_crn import DCCRN
from model.component.emotion_path.Wav2vec import Wav2VecModel
from model.component.text_path.SentenceModel import SentenceTransformerModel
from model.fusion_model import SimpleFusionModel
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter  # 增加 tensorboard 支持。
import sys


# CUR = os.path.dirname(__file__)
SOR_DIR = os.path.dirname(__file__)
checkpint_dir = os.path.join(SOR_DIR, "checkpoint")  # 模型权重保存目录
VISUALIZE_DIR = os.path.join(SOR_DIR, "visualize")  # 损失数据保存目录
visualize_train_dir = os.path.join(VISUALIZE_DIR, "train")
visualize_val_dir = os.path.join(VISUALIZE_DIR, "val")
visualize_test_dir = os.path.join(VISUALIZE_DIR, "test")
save_model_name = "model_v1"  # 模型权重文件名称
# SAVE_LOSS_NAME = ["mse_loss", "r2_score", "rmse_loss"]  # 用到的损失名称
SAVE_LOSS_NAME = ["bce_loss", "precision", "recall", "f1score"]
BATCH_SIZE = 1
START_LEARNING_RATE = 1e-4
LR_MILESTONES = [10, 20, 30, 50, 80, 100, 200]

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

writer = SummaryWriter("./tensorboard_logs")


def train_single_modal(
    model: torch.nn.Module = None,
    text_path: bool = False,
    bi_label: bool = False,
    resample: bool = True,
    resample_rate: int = 8000,
    concat_num: int = 3,
    load_epoch: int = 0,
    end_epoch: int = 100,
    save_interval: int = 10,
    checkpint_dir: str = None,
    save_model_name: str = None,
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
            get_text_ndarray(
                train=True,
                bi_label=bi_label,
                resample=resample,
                resample_rate=resample_rate,
            )
        )
    else:
        data_list_train, label_list_train, data_list_val, label_list_val = (
            get_waveform_ndarary(
                train=True,
                bi_label=bi_label,
                resample=resample,
                resample_rate=resample_rate,
                concat_num=concat_num,
            )
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
    BCE_Loss = []
    PRECISION_SCORE = []
    RECALL_SCORE = []
    F1_SCORE = []
    VAL_BCE_Loss = []
    VAL_PRECISION_SCORE = []
    VAL_RECALL_SCORE = []
    VAL_F1_SCORE = []
    VAL_BCE_Loss = []
    cur_epoch = 0

    try:
        # main training logic
        size = len(data_list_train)
        for eph in range(load_epoch + 1, end_epoch + 1):
            groundtruth = []
            predict = []
            print(f"epoch:{eph} ", "*" * 100)
            epoch_bce = 0.0
            for index, (x, y) in enumerate(zip(data_list_train, label_list_train)):
                groundtruth.append(1 if y == 1 else 0)
                if not text_path:
                    x = (
                        torch.unsqueeze(torch.tensor(x).to(device), dim=0)
                        if not isinstance(x, torch.Tensor)
                        else x.to(device)
                    )
                y = torch.unsqueeze(
                    torch.tensor(y, dtype=torch.float32).to(device), dim=0
                )
                y_hat = model(x)

                predict.append(1 if nn.Sigmoid()(y_hat).item() >= 0.53 else 0)
                bce_loss = bce_loss_fn(
                    torch.unsqueeze(y_hat, dim=0) if text_path else y_hat, y
                )

                bce_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # record loss
                current = index + 1
                bce_loss = bce_loss.item()
                epoch_bce += bce_loss
                print(f"bce_loss = {bce_loss:>12f}---[{current:>5d}/{size:>5d}]")

            epoch_bce /= size
            BCE_Loss.append(epoch_bce)
            print(f"train bce_lost = {epoch_bce:>12f}")
            scheduler.step()

            # 进行分类计算
            y_pred = np.array(predict)
            y_true = np.array(groundtruth)
            p = precision_score(y_true, y_pred, average="binary")
            r = recall_score(y_true, y_pred, average="binary")
            f1 = f1_score(y_true, y_pred, average="binary")
            PRECISION_SCORE.append(p)
            RECALL_SCORE.append(r)
            F1_SCORE.append(f1)
            print(f"train precision = {p}, recall = {r}, f1score = {f1}")

            # ===============================================================================================================================
            # validation logic
            model.eval()
            num_batches = len(data_list_val)
            val_bce_loss = 0.0
            predict = []
            groundtruth = []
            with torch.no_grad():
                for x, y in zip(data_list_val, label_list_val):
                    groundtruth.append(1 if y == 1 else 0)
                    if not text_path:
                        x = (
                            torch.unsqueeze(torch.tensor(x).to(device), dim=0)
                            if not isinstance(x, torch.Tensor)
                            else x.to(device)
                        )
                    y = torch.unsqueeze(
                        torch.tensor(y, dtype=torch.float32).to(device), dim=0
                    )

                    y_hat = model(x)

                    predict.append(1 if nn.Sigmoid()(y_hat).item() >= 0.53 else 0)
                    bce_loss = bce_loss_fn(
                        torch.unsqueeze(y_hat, dim=0) if text_path else y_hat, y
                    )

                    val_bce_loss += bce_loss.item()
            val_bce_loss /= num_batches
            VAL_BCE_Loss.append(val_bce_loss)
            print(f"val  bce_loss = {val_bce_loss:>12f}")

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

            VAL_PRECISION_SCORE.append(p)
            VAL_RECALL_SCORE.append(r)
            VAL_F1_SCORE.append(f1score)
            print(f"precision = {p}, recall = {r}, f1score = {f1score}")
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

            train_loss_save_dir = os.path.join(visualize_train_dir, save_model_name)
            val_loss_save_dir = os.path.join(visualize_val_dir, save_model_name)

            if not os.path.exists(train_loss_save_dir):
                os.mkdir(train_loss_save_dir)

            if not os.path.exists(val_loss_save_dir):
                os.mkdir(val_loss_save_dir)

            np.savez(
                os.path.join(
                    train_loss_save_dir,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=BCE_Loss,
            )

            np.savez(
                os.path.join(
                    val_loss_save_dir,
                    SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_BCE_Loss,
            )

            np.savez(
                os.path.join(
                    train_loss_save_dir,
                    SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=PRECISION_SCORE,
            )

            np.savez(
                os.path.join(
                    val_loss_save_dir,
                    SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_PRECISION_SCORE,
            )

            np.savez(
                os.path.join(
                    train_loss_save_dir,
                    SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=RECALL_SCORE,
            )

            np.savez(
                os.path.join(
                    val_loss_save_dir,
                    SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_RECALL_SCORE,
            )

            np.savez(
                os.path.join(
                    train_loss_save_dir,
                    SAVE_LOSS_NAME[3] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=F1_SCORE,
            )

            np.savez(
                os.path.join(
                    val_loss_save_dir,
                    SAVE_LOSS_NAME[3] + f"_epoch_to_{str(cur_epoch)}",
                ),
                data=VAL_F1_SCORE,
            )

            print(f"save success")
        else:
            print(f"save cancel, discard data")


def test_single_modal(
    model: nn.Module = None,
    text_path: bool = False,
    bi_label: bool = False,
    resample: bool = True,
    resample_rate: int = 8000,
    concat_num: int = 3,
    to_epoch: int = 0,
    checkpint_dir: str = None,
    save_model_name: str = None,
):

    print(f"use {device}")
    data_list_test, label_list_test = (
        get_text_ndarray(
            train=False,
            bi_label=bi_label,
            resample=resample,
            resample_rate=resample_rate,
            concat_num=concat_num,
        )
        if text_path
        else get_waveform_ndarary(
            train=False,
            bi_label=bi_label,
            resample=resample,
            resample_rate=resample_rate,
            concat_num=concat_num,
        )
    )
    dataset_len = len(data_list_test)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    BCE_Loss = []
    PRECISION_SCORE = []
    RECALL_SCORE = []
    F1_SCORE = []

    for cur_epoch in range(to_epoch):
        test_bce_loss = 0.0

        predict = []
        groundtruth = []

        model.load_state_dict(
            torch.load(
                os.path.join(checkpint_dir, save_model_name + f"_{cur_epoch + 1}")
            )
        )
        print(f"parameters loaded")
        model.to(device)
        model.eval()
        with torch.no_grad():
            for x, y in zip(data_list_test, label_list_test):

                groundtruth.append(1 if y == 1 else 0)

                x, y = (
                    x
                    if text_path
                    else torch.unsqueeze(torch.tensor(x), dim=0).to(device)
                ), torch.unsqueeze(
                    torch.tensor(y, dtype=torch.float32).to(device), dim=0
                )

                y_hat = model(x)

                predict.append(1 if nn.Sigmoid()(y_hat).item() >= 0.53 else 0)

                bce_loss = bce_loss_fn(
                    torch.unsqueeze(y_hat, dim=0) if text_path else y_hat, y
                )
                test_bce_loss += bce_loss.item()
            test_bce_loss /= dataset_len
            print(f"test  bce_loss = {test_bce_loss:>12f}")
            BCE_Loss.append(test_bce_loss)

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

        print(
            f"epoch: {cur_epoch + 1} : precision = {p}, recall = {r}, f1score = {f1score}"
        )
        PRECISION_SCORE.append(p)
        RECALL_SCORE.append(r)
        F1_SCORE.append(f1score)

    # save test loss
    # 创建对应的损失文件夹
    if not os.path.exists(visualize_test_dir):
        os.mkdir(visualize_test_dir)

    save_dir = os.path.join(visualize_test_dir, save_model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[0] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=BCE_Loss,
    )

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[1] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=PRECISION_SCORE,
    )

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[2] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=RECALL_SCORE,
    )

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[3] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=F1_SCORE,
    )

    print(f"save success")


def train_fusion_model(
    model: torch.nn.Module = None,
    resample_rate: int = 8000,
    load_epoch: int = 0,
    end_epoch: int = 100,
    save_interval: int = 10,
    checkpint_dir: str = None,
    save_model_name: str = None,
    start_lr: float = 1e-4,
):
    """对简易的融合模型进行训练

    Args:
        model (torch.nn.Module, optional): _description_. Defaults to None.
        load_epoch (int, optional): _description_. Defaults to 0.
        end_epoch (int, optional): _description_. Defaults to 100.
        save_interval (int, optional): _description_. Defaults to 10.
        checkpint_dir (str, optional): _description_. Defaults to None.
        save_model_name (str, optional): _description_. Defaults to None.
        visualize_train_dir (str, optional): _description_. Defaults to None.
        visualize_val_dir (str, optional): _description_. Defaults to None.
        start_lr (float, optional): _description_. Defaults to 1e-4.
    """

    train_dataloader, val_dataloader = get_trimodal_dataloader(
        batch_size=BATCH_SIZE, resmaple_rate=resample_rate
    )

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
    cur_epoch = 0

    try:
        # main training logic
        size = len(train_dataloader)
        for eph in range(load_epoch + 1, end_epoch + 1):
            groundtruth = []
            predict = []
            print(f"epoch:{eph} ", "*" * 100)
            epoch_bce = 0.0
            for index, (w, t, y) in enumerate(train_dataloader):
                groundtruth.append(1 if y == 1 else 0)
                w, y = (
                    (torch.unsqueeze(torch.tensor(w).to(device), dim=0))
                    if not isinstance(w, torch.Tensor)
                    else torch.squeeze(w, dim=0).to(device)
                ), torch.unsqueeze(y, dim=0).to(device)
                y_hat = model(w, t[0])

                predict.append(1 if nn.Sigmoid()(y_hat).item() >= 0.53 else 0)
                bce_loss = bce_loss_fn(y_hat, y)
                bce_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # record loss
                current = index + 1
                bce_loss = bce_loss.item()
                epoch_bce += bce_loss
                print(f"bce_loss = {bce_loss:>12f} ---[{current:>5d}/{size:>5d}]")
            epoch_bce /= size
            print(f"train bce_loss = {epoch_bce:>12f}")
            scheduler.step()

            # 进行分类计算
            y_pred = np.array(predict)
            y_true = np.array(groundtruth)
            p = precision_score(y_true, y_pred, average="binary")
            r = recall_score(y_true, y_pred, average="binary")
            f1 = f1_score(y_true, y_pred, average="binary")

            # 进行数据记录与可视化
            writer.add_scalars(
                main_tag="train",
                tag_scalar_dict={
                    "bce_loss": epoch_bce,
                    "precision": p,
                    "recall": r,
                    "f1score": f1,
                },
                global_step=cur_epoch,
            )

            print(f"train precision = {p}, recall = {r}, f1score = {f1}")

            # ===============================================================================================================================
            # validation logic
            model.eval()
            num_batches = len(val_dataloader)
            val_bce_loss = 0.0
            predict = []
            groundtruth = []
            with torch.no_grad():
                for w, t, y in val_dataloader:
                    groundtruth.append(1 if y == 1 else 0)
                    w, y = (
                        torch.unsqueeze(torch.tensor(w).to(device), dim=0)
                        if not isinstance(w, torch.Tensor)
                        else torch.squeeze(w, dim=0).to(device)
                    ), torch.unsqueeze(y, dim=0).to(device)

                    y_hat = model(w, t[0])
                    predict.append(1 if nn.Sigmoid()(y_hat).item() >= 0.53 else 0)
                    bce_loss = bce_loss_fn(y_hat, y)

                    val_bce_loss += bce_loss.item()
            val_bce_loss /= num_batches
            print(f"val  bce_loss = {val_bce_loss:>12f}")

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

            writer.add_scalars(
                main_tag="val",
                tag_scalar_dict={
                    "bce_loss": val_bce_loss,
                    "precision": p,
                    "recall": r,
                    "f1score": f1score,
                },
                global_step=cur_epoch,
            )

            # print(f"precision = {p}, recall = {r}, f1score = {f1score}")
            model.train()

            # save model parameters each [save_interval] epochs
            if eph % save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpint_dir, save_model_name + f"_{str(eph)}"),
                )
            cur_epoch = eph

    finally:
        print("over")
        # print(f"Are you sure to save loss data? [y/n]")
        # y = input()
        # if y == "y" or y == "Y":
        #     print(f"saving loss data" + "*" * 20)

        #     # 创建对应的损失文件夹
        #     if not os.path.exists(visualize_train_dir):
        #         os.mkdir(visualize_train_dir)
        #     if not os.path.exists(visualize_val_dir):
        #         os.mkdir(visualize_val_dir)

        #     train_loss_save_dir = os.path.join(visualize_train_dir, save_model_name)
        #     val_loss_save_dir = os.path.join(visualize_val_dir, save_model_name)

        #     if not os.path.exists(train_loss_save_dir):
        #         os.mkdir(train_loss_save_dir)

        #     if not os.path.exists(val_loss_save_dir):
        #         os.mkdir(val_loss_save_dir)

        #     # np.savez(
        #     #     os.path.join(
        #     #         train_loss_save_dir,
        #     #         SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=BCE_Loss,
        #     # )

        #     # np.savez(
        #     #     os.path.join(
        #     #         val_loss_save_dir,
        #     #         SAVE_LOSS_NAME[0] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=VAL_BCE_Loss,
        #     # )

        #     # np.savez(
        #     #     os.path.join(
        #     #         train_loss_save_dir,
        #     #         SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=PRECISION_SCORE,
        #     # )

        #     # np.savez(
        #     #     os.path.join(
        #     #         val_loss_save_dir,
        #     #         SAVE_LOSS_NAME[1] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=VAL_PRECISION_SCORE,
        #     # )

        #     # np.savez(
        #     #     os.path.join(
        #     #         train_loss_save_dir,
        #     #         SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=RECALL_SCORE,
        #     # )

        #     # np.savez(
        #     #     os.path.join(
        #     #         val_loss_save_dir,
        #     #         SAVE_LOSS_NAME[2] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=VAL_RECALL_SCORE,
        #     # )

        #     # np.savez(
        #     #     os.path.join(
        #     #         train_loss_save_dir,
        #     #         SAVE_LOSS_NAME[3] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=F1_SCORE,
        #     # )

        #     # np.savez(
        #     #     os.path.join(
        #     #         val_loss_save_dir,
        #     #         SAVE_LOSS_NAME[3] + f"_epoch_to_{str(cur_epoch)}",
        #     #     ),
        #     #     data=VAL_F1_SCORE,
        #     )

        #     print(f"save success")
        # else:
        #     print(f"save cancel, discard data")


def test_fusion_model(
    model: nn.Module = None,
    binary_label: bool = True,
    resample: bool = True,
    resample_rate: int = 4000,
    to_epoch: int = 0,
    checkpint_dir: str = None,
    save_model_name: str = None,
):
    print(f"use {device}")
    waveform_list, text_list, label_list = get_raw_trimodal_ndarray_dataset(
        train=False,
        binary_label=binary_label,
        resample=resample,
        resample_rate=resample_rate,
    )

    dataset_len = len(waveform_list)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    BCE_Loss = []
    PRECISION_SCORE = []
    RECALL_SCORE = []
    F1_SCORE = []

    for cur_epoch in range(to_epoch):
        test_bce_loss = 0.0
        predict = []
        groundtruth = []
        model.load_state_dict(
            torch.load(
                os.path.join(checkpint_dir, save_model_name + f"_{cur_epoch + 1}")
            )
        )
        print(f"parameters loaded")
        model.to(device)
        model.eval()
        with torch.no_grad():
            for w, t, y in zip(waveform_list, text_list, label_list):

                groundtruth.append(1 if y == 1 else 0)

                w = (
                    torch.unsqueeze(torch.tensor(w), dim=0).to(device)
                    if not isinstance(w, torch.Tensor)
                    else w.to(device)
                )
                y = torch.unsqueeze(torch.tensor(y).to(device), dim=0)

                y_hat = model(w, t)

                predict.append(1 if nn.Sigmoid()(y_hat).item() >= 0.53 else 0)

                bce_loss = bce_loss_fn(y_hat, y)
                test_bce_loss += bce_loss.item()
            test_bce_loss /= dataset_len
            # print(
            #     f"test  mse_loss = {test_mse_loss:>12f}, rmse_loss = {test_rmse_loss:>12f}"
            # )
            print(f"test  bce_loss = {test_bce_loss:>12f}")
            BCE_Loss.append(test_bce_loss.item())

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
        PRECISION_SCORE.append(p)
        RECALL_SCORE.append(r)
        F1_SCORE.append(f1score)

    # save test loss
    # 创建对应的损失文件夹
    if not os.path.exists(visualize_test_dir):
        os.mkdir(visualize_test_dir)

    save_dir = os.path.join(visualize_test_dir, save_model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[0] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=BCE_Loss,
    )

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[1] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=PRECISION_SCORE,
    )

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[2] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=RECALL_SCORE,
    )

    np.savez(
        os.path.join(
            save_dir,
            SAVE_LOSS_NAME[3] + f"_epoch_to_{str(to_epoch)}",
        ),
        data=F1_SCORE,
    )

    print(f"save success")


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

    # # load DCCRN
    # model = DCCRN(
    #     rnn_units=256,
    #     use_clstm=True,
    #     kernel_num=[32, 64, 128, 256, 256, 256],
    # )

    # # load Wav2Vec
    # model = Wav2VecModel()

    # # load SentenceTransformer
    # model = SentenceTransformerModel(device=device)

    # train_single_modal(
    #     model=model,
    #     text_path=True,
    #     bi_label=True,
    #     resample=True,
    #     resample_rate=8000,
    #     concat_num=3,
    #     load_epoch=0,
    #     end_epoch=50,
    #     save_interval=1,
    #     checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/SentenceTransformer_resample_augementation",
    #     save_model_name="SentenceTransformer_resample_augementation",
    #     start_lr=1e-3,
    # )

    # test_single_modal(
    #     model=model,
    #     text_path=True,
    #     bi_label=True,
    #     resample=True,
    #     resample_rate=8000,
    #     concat_num=3,
    #     to_epoch=50,
    #     checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/SentenceTransformer_resample_augementation",
    #     save_model_name="SentenceTransformer_resample_augementation",
    # )

    # load fusion model
    model = SimpleFusionModel(device=device)
    train_fusion_model(
        model=model,
        resample_rate=8000,
        load_epoch=0,
        end_epoch=20,
        save_interval=1,
        checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/fusion_model_resample_augmentation",
        save_model_name="fusion_model_resample_augmentation",
        start_lr=1e-5,
    )

    # test_fusion_model(
    #     model=model,
    #     binary_label=True,
    #     resample=True,
    #     resample_rate=4000,
    #     to_epoch=50,
    #     checkpint_dir="/public1/cjh/workspace/DepressionPrediction/checkpoint/fusion_model_resample_augmentation",
    #     save_model_name="fusion_model_resample_augmentation",
    # )
