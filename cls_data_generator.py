# from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/01-introduction-to-pytorch.html

import torch
import torch.utils.data as data


# データセットの定義
class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(
            low=0, high=2, size=(self.size, 2), dtype=torch.float32
        )  # highの値は含まない
        label = (data.sum(dim=1) == 1).to(
            torch.long
        )  # sum(dim=1): (size, 2) -> (size, 1), .to([型])は[型]に変換
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)  # ノイズ付加

        self.data = data
        self.label = label

    def __len__(self):  # データ数を戻す関数
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):  # idx番目のデータを返す関数
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]  # (1, 2)
        data_label = self.label[idx]
        return data_point, data_label
