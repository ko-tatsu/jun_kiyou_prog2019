import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
import chainer.functions as F

from sklearn.datasets import load_iris

# Iris データセットの読み込み
dataset = load_iris()

# 入力値と目標値を別々の変数へ格納
x = dataset.data
t = dataset.target

# Chainer がデフォルトで用いる float32 型へ変換
x = np.array(x, np.float32)
t = np.array(t, np.int32)

from chainer.datasets import TupleDataset

# 入力値と目標値を引数に与え、`TupleDataset` オブジェクトを作成
dataset = TupleDataset(x, t)

from chainer.datasets import split_dataset_random

n_train = int(len(dataset) * 0.7)
n_valid = int(len(dataset) * 0.1)

train, valid_test = split_dataset_random(dataset, n_train, seed=0)
valid, test = split_dataset_random(valid_test, n_valid, seed=0)

print('Training dataset size:', len(train))
print('Validation dataset size:', len(valid))
print('Test dataset size:', len(test))

from chainer import iterators

batchsize = 32

train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, shuffle=False, repeat=False)

class MLP(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=3):
        super().__init__()

        with self.init_scope():
            self.fc1 = L.Linear(None, n_mid_units)
            self.fc2 = L.Linear(n_mid_units, n_mid_units)
            self.fc3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h
'''
# 1. データセットからミニバッチを作成
train_batch = train_iter.next()
x, t = concat_examples(train_batch)

# 2. 順伝播（forward）の計算
y = net(x)

# 3. 損失（loss）の計算
loss = F.softmax_cross_entropy(y, t)

# 4. 逆伝播（backward）の計算
net.cleargrads()
loss.backward()

# 5. オプティマイザによってパラメータを更新
optimizer.update()
'''
from chainer import optimizers
from chainer import training

# ネットワークを作成
predictor = MLP()

# L.Classifier でラップし、損失の計算などをモデルに含める
net = L.Classifier(predictor)

# 最適化手法を選択してオプティマイザを作成し、最適化対象のネットワークを持たせる
optimizer = optimizers.MomentumSGD(lr=0.1).setup(net)

# アップデータにイテレータとオプティマイザを渡す
updater = training.StandardUpdater(train_iter, optimizer, device=-1) # device=-1でCPUでの計算実行を指定

trainer = training.Trainer(updater, (30, 'epoch'), out='results/iris_result1')

from chainer.training import extensions

trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name='log'))
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.Evaluator(valid_iter, net, device=-1), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'fc1/W/data/mean', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['fc1/W/grad/mean'], x_key='epoch', file_name='mean.png'))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.ParameterStatistics(net.predictor.fc1, {'mean': np.mean}, report_grads=True))

trainer.run()