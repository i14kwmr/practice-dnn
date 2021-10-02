import torch.nn as nn


# __init__関数では層を重ねて記述する．
# foward関数では入力層から出力層までの流れを__init__関数で定義したものを使って計算する．
# 出力層でsigmoid関数を使わない理由は予測値を使いたいから（損失関数の計算に必要）
class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x
