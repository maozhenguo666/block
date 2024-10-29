import numpy as np

# 加载一维标签数据
labels = np.load('int64.npy')

# 初始化一个二维数组，shape为(16432, 155)
num_samples = labels.shape[0]
num_classes = 155
one_hot_labels = np.zeros((num_samples, num_classes), dtype=np.float64)

# 循环将每个标签转为one-hot编码
for i in range(num_samples):
    one_hot_labels[i, labels[i]] = 1.0

# 保存为新的.npy文件
np.save('ntutestB_one_hot.npy', one_hot_labels)