import numpy as np

# 加载 .npy 文件

# x_test = np.load('test_joint_B_former.npy')
# y_test = np.load('int64.npy')
#
# # 保存为 .npz 文件
# np.savez('V1testformer.npz',  x_test=x_test, y_test=y_test)

# x_test = np.load('val_3.npy')
# y_test = np.load('vallabel.npy')
#
# # 保存为 .npz 文件
# np.savez('val.npz',  x_test=x_test, y_test=y_test)

x_test = np.load('test_3.npy')
y_test = np.load('testlabel.npy')

# 保存为 .npz 文件
np.savez('test.npz',  x_test=x_test, y_test=y_test)