import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os.path import join

DATASET_RAW_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/EATD-Corpus"
TRAIN_DATASET_DIR = join(DATASET_RAW_DIR, "train")
VAL_DATASET_DIR = join(DATASET_RAW_DIR, "validation")

NDARRAY_DIR = "/public1/cjh/workspace/DepressionPrediction/dataset/raw_ndarray"
NDARRAY_TRAIN_DIR = join(NDARRAY_DIR, "train")


class WaveformTFDataset(Dataset):
    def __init__(self, label_dataset: np.ndarray = None):
        super(WaveformTFDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "waveform_tf_raw.npz"))["arr_0"]
        self.label = label_dataset

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index], self.label[index]


class EmotinoHiddVecDataset(Dataset):
    def __init__(self, label_dataset: np.ndarray = None):
        super(EmotinoHiddVecDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "emotion_hiddenvec_raw.npz"))[
            "arr_0"
        ]
        self.label = label_dataset

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index], self.label[index]


class TextVecDataset(Dataset):
    def __init__(self, label_dataset: np.ndarray = None):
        super(TextVecDataset, self).__init__()
        self.data_np = np.load(join(NDARRAY_TRAIN_DIR, "text_raw.npz"))["arr_0"]
        self.label = label_dataset

    def __len__(
        self,
    ):
        return len(self.data_np)

    def __getitem__(self, index):
        return self.data_np[index], self.label[index]


class TriModalDataset(Dataset):
    """元素顺序: e, t, w, l

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
    ):
        super(TriModalDataset, self).__init__()
        self.emotion_np = np.load(join(NDARRAY_TRAIN_DIR, "emotion_hiddenvec_raw.npz"))[
            "arr_0"
        ]

        self.text_np = np.load(join(NDARRAY_TRAIN_DIR, "text_raw.npz"))["arr_0"]
        self.waveform_tf_np = np.load(join(NDARRAY_TRAIN_DIR, "waveform_tf_raw.npz"))[
            "arr_0"
        ]

        self.label = np.load(join(NDARRAY_TRAIN_DIR, "labels.npz"))["arr_0"] / 100

    def __len__(
        self,
    ):
        return len(self.emotion_np)

    def __getitem__(self, index):
        return (
            self.emotion_np[index],
            self.text_np[index],
            self.waveform_tf_np[index],
            self.label[index],
        )


# def get_tri_train_val_dataloader(batch_size: int = 32):
#     label_dataset = np.load(join(NDARRAY_TRAIN_DIR, "labels.npz"))["arr_0"]
#     waveform_tf_dataset = WaveformTFDataset(label_dataset)
#     emotion_hidd_vec_dataset = EmotinoHiddVecDataset(label_dataset)
#     text_vec_dataset = TextVecDataset(label_dataset)

#     waveform_tf_dataset_train, waveform_tf_dataset_test = torch.utils.data.random_split(
#         waveform_tf_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
#     )

#     emotion_hidd_vec_dataset_train, emotion_hidd_vec_dataset_test = (
#         torch.utils.data.random_split(
#             emotion_hidd_vec_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
#         )
#     )

#     text_vec_dataset_train, text_vec_dataset_test = torch.utils.data.random_split(
#         text_vec_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
#     )

#     waveform_tf_train_dataloader = DataLoader(
#         waveform_tf_dataset_train, batch_size=batch_size, shuffle=True
#     )
#     waveform_tf_test_dataloader = DataLoader(
#         waveform_tf_dataset_test, batch_size=batch_size, shuffle=True
#     )

#     emotion_hidd_vec_train_dataloader = DataLoader(
#         emotion_hidd_vec_dataset_train, batch_size=batch_size, shuffle=True
#     )
#     emotion_hidd_vec_test_dataloader = DataLoader(
#         emotion_hidd_vec_dataset_test, batch_size=batch_size, shuffle=True
#     )
#     text_vec_train_dataloader = DataLoader(
#         text_vec_dataset_train, batch_size=batch_size, shuffle=True
#     )
#     text_vec_test_dataloader = DataLoader(
#         text_vec_dataset_test, batch_size=batch_size, shuffle=True
#     )

#     return (
#         waveform_tf_train_dataloader,
#         waveform_tf_test_dataloader,
#         emotion_hidd_vec_train_dataloader,
#         emotion_hidd_vec_test_dataloader,
#         text_vec_train_dataloader,
#         text_vec_test_dataloader,
#     )


def get_tri_modal_dataloader(batch_size: int = 32):
    tri_modal_dataset = TriModalDataset()
    tri_modal_dataset_train, tri_modal_dataset_test = torch.utils.data.random_split(
        tri_modal_dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
    )

    tri_modal_dataloader_train = DataLoader(
        tri_modal_dataset_train, batch_size=batch_size, shuffle=True
    )
    tri_modal_dataloader_test = DataLoader(
        tri_modal_dataset_test, batch_size=batch_size, shuffle=False
    )
    return tri_modal_dataloader_train, tri_modal_dataloader_test


if __name__ == "__main__":
    # (
    #     waveform_tf_train_dataloader,
    #     waveform_tf_test_dataloader,
    #     emotion_hidd_vec_train_dataloader,
    #     emotion_hidd_vec_test_dataloader,
    #     text_vec_train_dataloader,
    #     text_vec_test_dataloader,
    # ) = get_tri_train_val_dataloader()

    # train_data, train_label = next(iter(waveform_tf_train_dataloader))
    # print(train_data.shape)
    # print(train_label.shape)

    dataset = TriModalDataset()
    e, t, w, l = dataset[0]
    print(e.shape)
    print(t.shape)
    print(w.shape)
    print(l.shape)

    train_input, train_label = get_tri_modal_dataloader()
    e, t, w, l = next(iter(train_input))
    print(e.shape)
    print(t.shape)
    print(w.shape)
    print(l.shape)
