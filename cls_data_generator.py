# from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/01-introduction-to-pytorch.html

import torch
import torch.utils.data as data


# データセットの定義
class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        """
        Inputs:
        - size: Number of data points we want to generate
        - std: Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):  # データとラベルを作成する関数

        data = torch.randint(
            low=0, high=2, size=(self.size, 2), dtype=torch.float32
        )  # 「0」or「1」
        label = (data.sum(dim=1) == 1).to(torch.long)  # .to([型])は[型]にキャスト
        data += self.std * torch.randn(data.shape)  # 問題を複雑にするためにノイズを付加

        self.data = data
        self.label = label

    def __len__(self):  # len(XORDataset)で呼び出される関数

        return self.size  # self.data.shape[0] or self.label.shape[0]

    def __getitem__(self, idx):  # idx番目のデータとラベルを返す関数（DataLoaderで呼び出される）

        data_point = self.data[idx]
        data_label = self.label[idx]

        return data_point, data_label
