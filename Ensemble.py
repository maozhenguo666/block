import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype = str)
    for idx, name in enumerate(val_txt):
        label = int(name.split('A')[1][:3])
        true_label.append(label)

    true_label = torch.from_numpy(np.array(true_label))
    return true_label


if __name__ == "__main__":
    


    j_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/skmixf__V1_J_3D/epoch1_test_score.pkl'
    b_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/skmixf__V1_B_3D/epoch1_test_score.pkl'
    jm_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/skmixf__V1_JM_3D/epoch1_test_score.pkl'
    bm_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/skmixf__V1_BM_3D/epoch1_test_score.pkl'

    j_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_J_3D/epoch1_test_score.pkl'
    b_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_B_3D/epoch1_test_score.pkl'
    jm_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_JM_3D/epoch1_test_score.pkl'
    bm_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_BM_3D/epoch1_test_score.pkl'
    
    j_file01 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/stt__V1_J_3D/epoch1_test_score.pkl'
    #b_file01 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/stt__V1_B_3D/epoch1_test_score.pkl'

    j_file11 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/hdgcn_V1_J_3D/epoch1_test_score.pkl'
    b_file11 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/hdgcn_V1_B_3D/epoch1_test_score.pkl'

    # 文件路径定义
    j_file2 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/blockgcn_V1_J_3D/epoch1_test_score.pkl'
    b_file2 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/blockgcn_V1_B_3D/epoch1_test_score.pkl'
    jm_file2 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/blockgcn_V1_JM_3D/epoch1_test_score.pkl'
    bm_file2 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/blockgcn_V1_BM_3D/epoch1_test_score.pkl'
    
    j3d_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/ctrgcn_V1_J_3D/epoch1_test_score.pkl'
    b3d_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/ctrgcn_V1_B_3D/epoch1_test_score.pkl'
    jm3d_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/ctrgcn_V1_JM_3D/epoch1_test_score.pkl'
    bm3d_file = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/ctrgcn_V1_BM_3D/epoch1_test_score.pkl'

    j_file3 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/tdgcn_V1_J_3D/epoch1_test_score.pkl'
    b_file3 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/tdgcn_V1_B_3D/epoch1_test_score.pkl'
    jm_file3 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/tdgcn_V1_JM_3D/epoch1_test_score.pkl'
    bm_file3 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/tdgcn_V1_BM_3D/epoch1_test_score.pkl'

    j_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_J_3D/epoch1_test_score.pkl'
    b_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_B_3D/epoch1_test_score.pkl'
    jm_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_JM_3D/epoch1_test_score.pkl'
    bm_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_BM_3D/epoch1_test_score.pkl'

    # 集成的文件
    File = [
        j_file, b_file, jm_file, bm_file,
        # b_file0, jm_file0, bm_file0,
        # j_file01, b_file01,
        j_file01,
        j_file11,b_file11,
        j_file2, b_file2, jm_file2, bm_file2,
        j3d_file, b3d_file, jm3d_file, bm3d_file,
        j_file3, b_file3, jm_file3, bm_file3,
        j_file4, b_file4, jm_file4, bm_file4
    ]

    
    # 设置与文件相对应的权重
    Numclass = 155
    Sample_Num = 4599  # 样本数为4599
    

    Rate = [1.2, 1.2, 0.2, 0.2,  
        1.2,             
        1.2,1.2,   
        1.2,1.2, 0.6, 0.2,    
        1.2, 1.2, 0.6, 0.2,    
        1.2, 1.2, 0.2, 0.5, 
        1.2, 1.2, 0.8, 0.2              
    ]
    

    

    
    # 计算得分
    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)

    # 保存结果
    np.save('pred.npy', final_score.numpy())

    print("Final score saved to 'final_score.npy'")

