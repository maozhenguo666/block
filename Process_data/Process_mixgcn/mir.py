import numpy as np

# 加载 .npy 文件

x_test = np.load('testbntmvc.npy')
y_test = np.load('int64.npy')

# 保存为 .npz 文件
np.savez('V1test.npz',  x_test=x_test, y_test=y_test)