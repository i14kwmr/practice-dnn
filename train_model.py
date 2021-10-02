import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm  # Progress bar

from cls_data_generator import XORDataset
from cls_model import SimpleClassifier


def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):

    # GPUが割り当て可能 -> GPUを割り当て，GPUが割り当て不可能 -> CPUを割り当て
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device", device)

    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:  # バッチごとのデータとラベル

            # Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(
                dim=1
            )  # Output is [Batch size, 1], but we want [Batch size]

            # Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())

            # Step 4: Perform backpropagation
            optimizer.zero_grad()  # 勾配を0にする．

            # Perform backpropagation
            loss.backward()  # 誤差逆伝播

            # Step 5: Update the parameters
            optimizer.step()  # 勾配を計算する．なお，上書きではなく，加算される．


if __name__ == "__main__":

    # 入力層2，隠れ層4，出力層1のモデルを初期化（モデルにより記述が異なる）
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)

    # Lossモジュールは提供されたものを用いる（代替可能）
    loss_module = nn.BCEWithLogitsLoss()

    # Input to the optimizer are the parameters of the model: model.parameters()
    # lr: learning rate（学習率）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # データセット作成
    train_dataset = XORDataset(size=1000)
    # データを読み込むLoaderの作成
    train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # train_modelの引数はmodel，modelを引数にして定義したoptimizer，定義したDataLoader，loss関数
    train_model(model, optimizer, train_data_loader, loss_module)

    # torch.save(object, filename). For the filename, any extension can be used
    state_dict = model.state_dict()  # 各層のパラメータが保存されている．
    torch.save(state_dict, "./output/our_model.tar")
