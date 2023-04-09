from model import NeuralNetwork
from best_args import load_data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

train_img, train_label, test_img, test_label = load_data(download=False,
                                                         flatten=True,
                                                         normalize=True)
# 导入模型
network = NeuralNetwork.load("1.model")
# 导出分类精度
w1, w2, b1, b2 = network.get_params()

# 每个隐藏层和所有输入层连接权重对应一个和输入图片一样28*28的灰度图片
for i in range(20):
    img = w1[:, i].reshape(28, 28) * 255
    img = Image.fromarray(np.uint8(img))
    plt.imshow(img, cmap='gray')
    plt.savefig('参数/'+str(i)+'.png')

x1 = plt.subplot(121)
img1 = Image.fromarray(np.uint8(w1 * 255))
plt.title('w1')
plt.imshow(img1, cmap='gray')
x2 = plt.subplot(122)
img2 = Image.fromarray(np.uint8(w2 * 255))
plt.title('w2')
plt.imshow(img2, cmap='gray')
plt.savefig('网络参数可视化', dpi=200)
plt.show()
print("在测试集上预测的分类精度acc=", network.accuracy(test_img, test_label))
