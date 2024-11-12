import numpy as np

# 加载数据
data = np.load('G:/无人机/数据处理/基于无人机的人体行为识别-参赛资源(省赛)/dataguo/data/test_joint.npy')

# 获取原始数据的形状
N, C, T, V, M = data.shape

# 初始化新的形状
new_shape = (N, T, M * V * C)
transformed_data = np.zeros(new_shape, dtype=np.float32)

# 遍历每个样本和每一帧
for n in range(N):
    for t in range(T):
        for m in range(M):
            for v in range(V):
                for c in range(C):
                    # 计算新的索引并赋值
                    new_index = m * V * C + v * C + c
                    transformed_data[n, t, new_index] = data[n, c, t, v, m]

# 保存新数据
np.save('test_3.npy', transformed_data)
