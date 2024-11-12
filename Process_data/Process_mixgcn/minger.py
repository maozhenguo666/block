import numpy as np

# 加载 .npy 文件
x_train = np.load('train_joint_trans.npy')
y_train = np.load('G:/无人机/数据处理/基于无人机的人体行为识别-参赛资源(省赛)/dataguo/data/train_label.npy')
x_test = np.load('val_joint_trans.npy')
y_test = np.load('G:/无人机/数据处理/基于无人机的人体行为识别-参赛资源(省赛)/dataguo/data/val_label.npy')

# 保存为 .npz 文件
np.savez('train.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
