import torch
import torch.utils.data as data

from cls_data_generator import XORDataset
from cls_model import SimpleClassifier


# modelをevalモードにすることが必要
def test_model(model, data_loader):

    # deviceという概念
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    model.eval()  # Set model to eval mode, evalモードにする
    true_preds, num_preds = 0.0, 0.0  # (true_preds=TP+TN, num_preds=TP+TN+FP+FN)

    with torch.no_grad():  # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set, テストではラベルに計算し直して合っているか検証する
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds)  # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long()  # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")


if __name__ == "__main__":

    # Load state dict from the disk (make sure it is the same name as above)
    state_dict = torch.load("our_model.tar")  # torch.load: 保存したパラメータを読み込み(state)

    # Create a new model and load the state
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)  # 新たなモデルの定義
    model.load_state_dict(state_dict)  # model.load_state_dict: モデルに保存したパラメータを読み込み

    test_dataset = XORDataset(size=500)  # テストデータセットの作成
    # drop_last -> Don't drop the last batch although it is smaller than 128, 最後までやり切る
    # dataloaderの定義はtrainとほとんど変わらない
    test_data_loader = data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, drop_last=False
    )

    test_model(model, test_data_loader)