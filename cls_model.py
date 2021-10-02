import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # ネットワークの初期化，層を登録する．
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # 予測をするため順番に層の計算を行う．
        # 出力を「1」or「0」に丸めないのは推定値が損失関数の計算に必要であるため
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x
