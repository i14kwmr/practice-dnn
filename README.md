# practice-dnn
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

1. [本リポジトリ](https://github.com/i14kwmr/practice-dnn)をclone
```
$ git clone git@github.com:i14kwmr/practice-dnn.git
```

2. 学習したパラメータを書き出せるように権限を変更（要注意，他の方法を探すべき）
```
$ chmod 0777 practice-dnn/
```

3. Dockerで環境を構築
```
$ cd practice-dnn
$ docker-compose up -d --build
$ docker compose exec python3 bash
```

4. ワークスペースに移動
```
$ cd practice-dnn
```

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
