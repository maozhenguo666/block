import numpy as np
import pickle
from tqdm import tqdm
from skopt import gp_minimize

# 加载模型得分文件和标签文件
score_files = [
    # 'G:/无人机/贝叶斯/mixformer/output/skmixf__V1_J/epoch1_test_score.pkl',
    # 'G:/无人机/贝叶斯/mixformer/output/skmixf__V1_B/epoch1_test_score.pkl',
    # 'G:/无人机/贝叶斯/mixformer/output/skmixf__V1_JM/epoch1_test_score.pkl',
    # 'G:/无人机/贝叶斯/mixformer/output/skmixf__V1_BM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/outputwe/output/hy__V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/outputwe/output/hy__V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/outputwe/output/hy__V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/outputwe/output/hy__V1_BM/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixformer/output/stt__V1_J/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixformer/output/stt__V1_B/epoch1_test_score.pkl',

    # 'G:/无人机/贝叶斯/MMCL/j/epoch1_test_score.pkl',
    # 'G:/无人机/贝叶斯/MMCL/b/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/MMCL/jm/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/MMCL/bm/epoch1_test_score.pkl',
    #
    'G:/无人机/贝叶斯/FR-Head+MMCL/j/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/FR-Head+MMCL/b/epoch1_test_score.pkl',
    # 'G:/无人机/贝叶斯/FR-Head+MMCL/jm/epoch1_test_score.pkl',
    # 'G:/无人机/贝叶斯/FR-Head+MMCL/bm/epoch1_test_score.pkl',

    'G:/无人机/贝叶斯/mixgcn/output/hdgcn_V1_J_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/hdgcn_V1_B_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/hdgcn_V1_JM_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/hdgcn_V1_BM_3D/epoch1_test_score.pkl',

    'G:/无人机/贝叶斯/mixgcn/output/frheadhdgcn_V1_J_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/frheadhdgcn_V1_B_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/frheadhdgcn_V1_JM_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/frheadhdgcn_V1_BM_3D/epoch1_test_score.pkl',



    'G:/无人机/贝叶斯/mixgcn/output/blockgcn_V1_J_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/blockgcn_V1_B_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/blockgcn_V1_JM_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/blockgcn_V1_BM_3D/epoch1_test_score.pkl',



    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+ctr/j/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+ctr/b/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/ctr/jm/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/ctr/bm/epoch1_test_score.pkl',


    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+tdgcn/j/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+tdgcn/b/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/tdgcn/jm/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/tdgcn/bm/epoch1_test_score.pkl',


    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+tcagcn/j/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+tcagcn/b/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+tcagcn/jm/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/FR-Head+tcagcn/bm/epoch1_test_score.pkl',

    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/outputwe/output/mstgcn_V1_J/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/outputwe/output/mstgcn_V1_B/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/outputwe/output/mstgcn_V1_JM/epoch1_test_score.pkl',
    # '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/outputwe/output/mstgcn_V1_BM/epoch1_test_score.pkl',

    'G:/无人机/贝叶斯/mixgcn/output/degcn_V1_J_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/degcn_V1_B_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/degcn_V1_JM_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/degcn_V1_BM_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/jbf_V1_J_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/jbf_V1_B_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/jbf_V1_JM_3D/epoch1_test_score.pkl',
    'G:/无人机/贝叶斯/mixgcn/output/jbf_V1_BM_3D/epoch1_test_score.pkl'
]

# 读取得分
scores = []
for file in score_files:
    with open(file, 'rb') as f:
        scores.append(np.array(list(pickle.load(f).values())))

# 加载标签
label = np.load('G:/无人机/贝叶斯/val_label.npy')  # 替换为你的标签文件路径


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


space = [(0.1, 1.5) for _ in range(len(scores))]
result = gp_minimize(objective, space, n_calls=200, random_state=0)

print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
print('Optimal weights: {}'.format(result.x))
