# practice-dnn
参考: PyTorch Lightning「[TUTORIAL 1: INTRODUCTION TO PYTORCH](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/01-introduction-to-pytorch.html)」
* Pytorchを用いたDNNの実装の練習（DNNでXORを実現）
  * (入力1, 入力2) -> 出力
    * (0, 0) -> 0
    * (0, 1) -> 1
    * (1, 0) -> 1
    * (1, 1) -> 0
  * 入力にはノイズを付加している（
  <a href="https://www.codecogs.com/eqnedit.php?latex=0&space;\pm&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?0&space;\pm&space;\epsilon" title="0 \pm \epsilon" /></a> 
  or 
  <a href="https://www.codecogs.com/eqnedit.php?latex=1\pm&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1\pm&space;\epsilon" title="1\pm \epsilon" /></a>）
  

## Usage

### 準備
1. [本リポジトリ](https://github.com/i14kwmr/practice-dnn)をclone
```
$ git clone git@github.com:i14kwmr/practice-dnn.git
```

2. フォルダに移動
```
$ cd practice-dnn
```

3. 学習したパラメータを書き出せるようにフォルダを作成し権限を変更
```
$ mkdir output
$ chmod 0777 output
```

4. Dockerで環境を構築しワークスペースに移動
```
$ docker-compose up -d --build
$ docker compose exec python3 bash
$ cd practice-dnn
```

### 実行
5. 学習
```
$ python train_model.py
```

6. 評価
```
$ python test_model.py
```
出力例
```
Accuracy of the model: 99.60%
```
