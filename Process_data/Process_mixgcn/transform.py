import numpy as np

# 加载数据
data = np.load('G:/无人机/数据处理/基于无人机的人体行为识别-参赛资源(省赛)/dataguo/data/test_joint.npy')

# 初始化新的数组
N, C, T, V, M = data.shape
new_data = np.zeros((N, M, T, V, C), dtype=np.float32)

# 标准化到 [-1, 1]
# data_min = data.min(axis=(1, 2, 3, 4), keepdims=True)
# data_max = data.max(axis=(1, 2, 3, 4), keepdims=True)
# data_std = (data - data_min) / (data_max - data_min) * 2 - 1

# 遍历每个样本和每个人的数据
for n in range(N):
    for m in range(M):
        for t in range(T):
            for v in range(V):
                # 提取每个人的数据（x, y, z）
                new_data[n, m, t, v, :] = data[n, :, t, v, m]

# 保存新数据
np.save('test_joint_trans.npy', new_data)
