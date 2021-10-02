import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm  # Progress bar

from cls_data_generator import XORDataset
from cls_model import SimpleClassifier


def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):

    # deviceという概念
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

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
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()  # 勾配を0にする
            # Perform backpropagation
            loss.backward()  # 誤差逆伝播

            # Step 5: Update the parameters
            optimizer.step()  # 勾配を計算する


if __name__ == "__main__":

    # 入力層2, 隠れ層4, 出力層1のモデルを初期化
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)

    # Lossモジュールも提供されたものを用いる．
    # Lossモジュールは現状modelと繋がっていない． trainで呼び出すようにする．
    loss_module = nn.BCEWithLogitsLoss()

    # Input to the optimizer are the parameters of the model: model.parameters()
    # lr: learning rate（学習率）
    # 引数はモデルのパラメータ, これをtrainで呼び出す．
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # modelのパラメータを引数としている

    # optimizer.zero_grad()は勾配を0にする
    # optimizer.step()で勾配を計算する．なお，上書きではなく，加算される．

    train_dataset = XORDataset(size=1000)  # 1000個のデータセット作成
    train_data_loader = data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )  # データを読み込むLoaderの作成

    # train_modelの引数はmodel, modelを引数にして定義したoptimizer, DataLoader, 定義したloss関数
    train_model(model, optimizer, train_data_loader, loss_module)

    # state_dictに各層のパラメータが保存されている
    state_dict = model.state_dict()
    print(state_dict)

    # torch.save(object, filename). For the filename, any extension can be used
    torch.save(state_dict, "our_model.tar")
