import numpy as np
import pickle
from tqdm import tqdm
from skopt import gp_minimize

# 加载模型得分文件和标签文件
score_files = [
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_J/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_B/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_JM/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_BM/epoch1_test_score.pkl',
    # './Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl',
    # './Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_J_3D/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_B_3D/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_JM_3D/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_BM_3D/epoch1_test_score.pkl'
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl'
]

# 读取得分
scores = []
for file in score_files:
    with open(file, 'rb') as f:
        scores.append(np.array(list(pickle.load(f).values())))

# 加载标签
label = np.load('/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/test_label_A.npy')  # 替换为你的标签文件路径

def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        r = sum(scores[j][i] * weights[j] for j in range(len(scores)))
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

space = [(0.2, 1.2) for _ in range(len(scores))]
result = gp_minimize(objective, space, n_calls=200, random_state=0)

print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
print('Optimal weights: {}'.format(result.x))
import numpy as np
import pickle
from tqdm import tqdm
from skopt import gp_minimize

# 加载模型得分文件和标签文件
score_files = [
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/skmixf__V1_BM/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_J/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_B/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_JM/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/hy__V1_BM/epoch1_test_score.pkl',
    # './Model_inference/Mix_Former/output/skmixf__V1_k2/epoch1_test_score.pkl',
    # './Model_inference/Mix_Former/output/skmixf__V1_k2M/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_J_3D/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_B_3D/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_JM_3D/epoch1_test_score.pkl',
    '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/blockgcn_V1_BM_3D/epoch1_test_score.pkl'
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/tdgcn_V1_BM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/mstgcn_V1_BM/epoch1_test_score.pkl'
]

# 读取得分
scores = []
for file in score_files:
    with open(file, 'rb') as f:
        scores.append(np.array(list(pickle.load(f).values())))

# 加载标签
label = np.load('/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/test_label_A.npy')  # 替换为你的标签文件路径

def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        r = sum(scores[j][i] * weights[j] for j in range(len(scores)))
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

space = [(0.2, 1.2) for _ in range(len(scores))]
result = gp_minimize(objective, space, n_calls=200, random_state=0)

print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
print('Optimal weights: {}'.format(result.x))
