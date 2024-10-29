import numpy as np

# 生成一个全为0的数组，数据类型为int64，长度为4599
zeros_array_int64 = np.zeros(4599, dtype=np.int64)

# 保存为npy文件
np.save("int64.npy", zeros_array_int64)


