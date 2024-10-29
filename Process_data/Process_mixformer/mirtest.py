import numpy as np

# 加载 .npy 文件

# x_test = np.load('test_joint_B_former.npy')
# y_test = np.load('int64.npy')
#
# # 保存为 .npz 文件
# np.savez('V1testformer.npz',  x_test=x_test, y_test=y_test)

x_test = np.load('ntutestB.npy')
y_test = np.load('ntutestB_one_hot.npy')

# 保存为 .npz 文件
np.savez('V1formertest.npz',  x_test=x_test, y_test=y_test)