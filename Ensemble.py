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
    


    j_file = 'G:/无人机/贝叶斯/scoreCC/ski/j/epoch1_test_score.pkl'
    b_file = 'G:/无人机/贝叶斯/scoreCC/ski/b/epoch1_test_score.pkl'
    jm_file = 'G:/无人机/贝叶斯/scoreCC/ski/jm/epoch1_test_score.pkl'
    bm_file = 'G:/无人机/贝叶斯/scoreCC/ski/bm/epoch1_test_score.pkl'
    #
    # j_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_J_3D/epoch1_test_score.pkl'
    # b_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_B_3D/epoch1_test_score.pkl'
    # jm_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_JM_3D/epoch1_test_score.pkl'
    # bm_file0 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_Former/output/output/hy__V1_BM_3D/epoch1_test_score.pkl'

    
    j_file01 = 'G:/无人机/贝叶斯/scoreCC/STTFormer/j/epoch1_test_score.pkl'
    b_file01 = 'G:/无人机/贝叶斯/scoreCC/STTFormer/b/epoch1_test_score.pkl'

    j_file02 = 'G:/无人机/贝叶斯/scoreCC/MMCL/j/epoch1_test_score.pkl'
    b_file02 = 'G:/无人机/贝叶斯/scoreCC/MMCL/b/epoch1_test_score.pkl'
    jm_file02 = 'G:/无人机/贝叶斯/scoreCC/MMCL/jm/epoch1_test_score.pkl'
    bm_file02 = 'G:/无人机/贝叶斯/scoreCC/MMCL/bm/epoch1_test_score.pkl'

    j_file03 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+MMCL/j/epoch1_test_score.pkl'
    b_file03 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+MMCL/b/epoch1_test_score.pkl'
    jm_file03 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+MMCL/jm/epoch1_test_score.pkl'
    bm_file03 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+MMCL/bm/epoch1_test_score.pkl'

    j_file11 = 'G:/无人机/贝叶斯/scoreCC/hdgcn/j/epoch1_test_score.pkl'
    b_file11 = 'G:/无人机/贝叶斯/scoreCC/hdgcn/b/epoch1_test_score.pkl'
    jm_file11 ='G:/无人机/贝叶斯/scoreCC/hdgcn/jm/epoch1_test_score.pkl'
    bm_file11 = 'G:/无人机/贝叶斯/scoreCC/hdgcn/bm/epoch1_test_score.pkl'

    j_file12 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+hdgcn/j/epoch1_test_score.pkl'
    b_file12 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+hdgcn/b/epoch1_test_score.pkl'
    jm_file12 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+hdgcn/jm/epoch1_test_score.pkl'
    bm_file12 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+hdgcn/bm/epoch1_test_score.pkl'
    # 文件路径定义
    j_file2 = 'G:/无人机/贝叶斯/scoreCC/block新/j/epoch1_test_score.pkl'
    b_file2 = 'G:/无人机/贝叶斯/scoreCC/block新/b/epoch1_test_score.pkl'
    jm_file2 = 'G:/无人机/贝叶斯/scoreCC/block新/jm/epoch1_test_score.pkl'
    bm_file2 = 'G:/无人机/贝叶斯/scoreCC/block新/bm/epoch1_test_score.pkl'
    
    j3d_file = 'G:/无人机/贝叶斯/scoreCC/FR-Head+ctr/j/epoch1_test_score.pkl'
    b3d_file = 'G:/无人机/贝叶斯/scoreCC/FR-Head+ctr/b/epoch1_test_score.pkl'
    jm3d_file = 'G:/无人机/贝叶斯/scoreCC/FR-Head+ctr/jm/epoch1_test_score.pkl'
    bm3d_file = 'G:/无人机/贝叶斯/scoreCC/FR-Head+ctr/bm/epoch1_test_score.pkl'

    j_file3 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+tdgcn/j/epoch1_test_score.pkl'
    b_file3 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+tdgcn/b/epoch1_test_score.pkl'
    jm_file3 = 'G:/无人机/贝叶斯/scoreCC/tdgcn/jm/epoch1_test_score.pkl'
    bm_file3 = 'G:/无人机/贝叶斯/scoreCC/tdgcn/bm/epoch1_test_score.pkl'

    j_file31 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+tcagcn/j/epoch1_test_score.pkl'
    b_file31 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+tcagcn/b/epoch1_test_score.pkl'
    jm_file31 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+tcagcn/jm/epoch1_test_score.pkl'
    bm_file31 = 'G:/无人机/贝叶斯/scoreCC/FR-Head+tcagcn/bm/epoch1_test_score.pkl'

    j_file4 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/degcn/j/epoch1_test_score.pkl'
    b_file4 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/degcn/b/epoch1_test_score.pkl'
    jm_file4 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/degcn/jm/epoch1_test_score.pkl'
    bm_file4 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/degcn/bm/epoch1_test_score.pkl'
    # j_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_J_3D/epoch1_test_score.pkl'
    # b_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_B_3D/epoch1_test_score.pkl'
    # jm_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_JM_3D/epoch1_test_score.pkl'
    # bm_file4 = '/mnt/workspace/ICMEW2024-Track10/Model_inference/Mix_GCN/output/output/mstgcn_V1_BM_3D/epoch1_test_score.pkl'
    j_file5 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/jbf/j/epoch1_test_score.pkl'
    b_file5 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/jbf/b/epoch1_test_score.pkl'
    jm_file5 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/jbf/jm/epoch1_test_score.pkl'
    bm_file5 = 'G:/无人机/贝叶斯/scoreCC/DEGCN/jbf/bm/epoch1_test_score.pkl'
    # 集成的文件
    # File = [
    
    File = [
        j_file01, b_file01,
        # j_file02, b_file02, jm_file02, bm_file02,
        # j_file03, b_file03, jm_file03, bm_file03,
        jm_file02, bm_file02,j_file03, b_file03,
        j_file11, b_file11, jm_file11, bm_file11,
        j_file12, b_file12, jm_file12, bm_file12,
        j_file2, b_file2, jm_file2, bm_file2,
        j3d_file, b3d_file, jm3d_file, bm3d_file,
        j_file3, b_file3, jm_file3, bm_file3,
        j_file31, b_file31, jm_file31, bm_file31,
        j_file4, b_file4, jm_file4, bm_file4,
        j_file5, b_file5, jm_file5, bm_file5
    ]
    
    # 设置与文件相对应的权重
    Numclass = 155
    Sample_Num = 4307

    

    Rate = [
        0.1, 1.5,  # STTFormer J ,B
        0.19060032933206877, 0.15, 0.3, 1.5,  # MMCL         JM,BM, J(FR-Head),B(FR-Head)
        1.5, 1.5, 1.5, 1.0553545584180628,  # HDGCN   J,B, JM,BM
        1.5, 1.5, 0.1, 1.5,  # HDGCN(FR-Head)   J,B, JM,BM
        0.7, 1.2, 0.1, 0.1,  # BlockGCN   J,B, JM,BM
        1.5, 1.5, 0.1, 0.1,  # CTRGCN J(FR-Head),B(FR-Head),JM,BM
        0.644289268983005, 1.5, 0.1, 0.1,  # TDGCN J(FR-Head),B(FR-Head),JM,BM

        1.3, 1.3, 0.3042270233860835, 0.1,  # TCAGCN(FR-Head)  J,B, JM,BM
        0.1, 0.1, 0.1, 0.1,  # DEGCN  J,B, JM,BM
        1.5, 1.5, 0.1, 0.1  # DEGCN(JBF) J,B, JM,BM
    ]

    # 计算得分
    final_score = Cal_Score(File, Rate, Sample_Num, Numclass)

    # 保存结果
    np.save('pred.npy', final_score.numpy())

    print("Final score saved to 'final_score.npy'")

