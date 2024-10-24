import numpy as np

# 加载 .npy 文件
x_train = np.load('train_joint_transformed2d.npy')
y_train = np.load('train_label.npy')
x_test = np.load('test_joint_A_2d.npy')
y_test = np.load('test_label_A.npy')

# 保存为 .npz 文件
np.savez('V12.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
