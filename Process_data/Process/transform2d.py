import numpy as np

# 加载数据
data = np.load('新建文件夹/testBSZ.npy')

# 初始化新的数组，目标维度为 (N, T, M, V, C)
N, C, T, V, M = data.shape
new_data = np.zeros((N, T, M, V, C), dtype=data.dtype)

# 遍历每个样本、每帧、每个人和每个节点的数据
for n in range(N):
    for t in range(T):
        for m in range(M):
            for v in range(V):
                # 提取每帧中每个人的 (x, y) 数据
                new_data[n, t, m, v, :] = data[n, :, t, v, m]

# 保存新数据
np.save('新建文件夹/testbntmvc.npy', new_data)
