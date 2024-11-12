import numpy as np

# 加载 .npy 文件

# x_test = np.load('val_joint_trans.npy')
# y_test = np.load('G:/无人机/数据处理/基于无人机的人体行为识别-参赛资源(省赛)/dataguo/data/val_label.npy')
#
# # 保存为 .npz 文件
# np.savez('val.npz',  x_test=x_test, y_test=y_test)

x_test = np.load('test_joint_trans.npy')
y_test = np.load('int64.npy')

# 保存为 .npz 文件
np.savez('test.npz',  x_test=x_test, y_test=y_test)