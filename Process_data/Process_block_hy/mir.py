import numpy as np

# 加载 .npy 文件
x_train = np.load('train_3.npy')
y_train = np.load('trainlabel.npy')
x_test = np.load('val_3.npy')
y_test = np.load('vallabel.npy')

# 保存为 .npz 文件
np.savez('train.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
