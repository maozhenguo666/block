import numpy as np

# 加载原始数据
data = np.load('test_joint_B.npy')

# 检查数据的形状（N, C, T, V, M），其中C为3代表XYZ通道
print("原始数据形状:", data.shape)

# 去掉Z通道（即C维度中的第3个通道）
# 假设C维度是第二个维度 (即C=3 -> X, Y, Z)，我们保留X, Y两个通道
data_2d = data[:, :2, :, :, :]  # 保留C维度中的前两个通道

# 检查新的二维数据形状
print("二维数据形状:", data_2d.shape)

# 保存处理后的数据
np.save('新建文件夹/testBSZ.npy', data_2d)
