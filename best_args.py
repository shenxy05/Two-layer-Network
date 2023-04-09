import numpy as np
import os
import gzip
import urllib
import errno
import matplotlib.pyplot as plt
from model import NeuralNetwork


def load_data(download=False, flatten=True, normalize=False, validation=False):
    # 下载数据集
    if download:
        folder = 'MNIST'
        # 如果文件夹不存在，创建文件夹
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urls = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        ]
        # 从链接中下载数据集在文件夹中
        for url in urls:
            print('Downloading ' + url)
            filename = url.rpartition('/')[2]
            urllib.request.urlretrieve(url, os.path.join(folder, filename))

    # 加载数据集到numpy
    with gzip.open('./MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as test_labels:
        test_label = np.frombuffer(test_labels.read(),
                                   dtype=np.uint8,
                                   offset=8)
    with gzip.open('./MNIST/t10k-images-idx3-ubyte.gz', 'rb') as test_imgs:
        test_img = np.frombuffer(test_imgs.read(), dtype=np.uint8,
                                 offset=16).reshape(-1, 28, 28)
    with gzip.open('./MNIST/train-labels-idx1-ubyte.gz', 'rb') as train_labels:
        train_label = np.frombuffer(train_labels.read(),
                                    dtype=np.uint8,
                                    offset=8)
    with gzip.open('./MNIST/train-images-idx3-ubyte.gz', 'rb') as train_imgs:
        train_img = np.frombuffer(train_imgs.read(), dtype=np.uint8,
                                  offset=16).reshape(-1, 28, 28)

    # 对label进行one-hot处理
    temp = np.zeros((test_label.size, 10))
    for i, row in enumerate(temp):
        row[test_label[i]] = 1
    test_label = temp
    temp = np.zeros((train_label.size, 10))
    for i, row in enumerate(temp):
        row[train_label[i]] = 1
    train_label = temp

    # 对输入图像进行归一化
    if normalize:
        train_img = train_img.astype(np.float32) / 255.0
        test_img = test_img.astype(np.float32) / 255.0

    # 将每张样本平铺到一维
    if flatten:
        test_img = test_img.reshape(-1, 784)
        train_img = train_img.reshape(-1, 784)

    # 从原训练集中划分验证集
    if validation:
        val_img = train_img[50000:]
        val_label = train_label[50000:]
        train_img = train_img[:50000]
        train_label = train_label[:50000]
        return train_img, train_label, val_img, val_label, test_img, test_label
    else:
        return train_img, train_label, test_img, test_label


if __name__ == '__main__':
    train_img, train_label, val_img, val_label, _, _ = load_data(normalize=True, validation=True)
    best_arg = {}
    acc = 0
    best_arg['lr'] = 0.1
    best_arg['lambda'] = 1e-4
    best_arg['hidden'] = 30
    for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        for lamb_da in [1e-4, 1e-5, 0]:
            for hidden in [30, 50, 100, 150]:
                network = NeuralNetwork(784, hidden, 10, lr=lr, lamb_da=lamb_da)
                epoch = 5
                print("-----learning rate =", lr, "-------")
                print("-----weight dacay =", lamb_da, "-------")
                print("-----hidden size =", hidden, "-------")
                train_loss = []
                train_acc = []
                val_loss = []
                val_acc = []
                for e in range(epoch):
                    iter = np.arange(len(train_label))
                    np.random.shuffle(iter)  # 随机选取样本计算梯度
                    batch_size = 100
                    batch_index = np.arange(0, len(train_label) + 1, batch_size)
                    count = 0
                    for i in range(len(batch_index) - 1):
                        # 用于更新模型的批量
                        index = iter[batch_index[i]:batch_index[i + 1]]
                        network.update(train_img[index], train_label[index])
                        if i % 60 == 0:
                            train_loss.append(network.loss(train_img, train_label))
                            train_acc.append(network.accuracy(train_img, train_label))
                            val_loss.append(network.loss(val_img, val_label))
                            val_acc.append(network.accuracy(val_img, val_label))
                    network.lr_decay(0.9)  # 学习率衰减

                # 画loss和acc曲线，保存图片
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
                plt.rcParams['axes.unicode_minus'] = False

                plt.figure(figsize=(12, 6), dpi=150)
                x1 = np.arange(len(train_loss))
                ax1 = plt.subplot(121)
                plt.plot(x1, train_loss, label='train')
                plt.plot(x1, val_loss, label='validation')
                plt.title('loss曲线')
                plt.xlabel("iteration")
                plt.ylabel("loss")
                plt.legend()

                x2 = np.arange(len(train_acc))
                ax2 = plt.subplot(122)
                plt.plot(x2, train_acc, label='train')
                plt.plot(x2, val_acc, label='validation')
                plt.title("accuracy曲线")
                plt.xlabel("iteration")
                plt.ylabel("accuracy")
                plt.legend()
                plt.savefig(str(lr) + '+' + str(lamb_da) + '+' + str(hidden) + '.png')
                plt.clf()
                print(network.accuracy(val_img, val_label))
                if network.accuracy(val_img, val_label) > acc:
                    best_arg['lr'] = lr
                    best_arg['lambda'] = lamb_da
                    best_arg['hidden'] = hidden
                    acc = network.accuracy(val_img, val_label)
                print("最佳参数:lr=", best_arg['lr'], ',labmda=', best_arg['lambda'], ',hidden=', best_arg['hidden'])
