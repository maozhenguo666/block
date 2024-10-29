下面是按你的需求生成和操作文件的步骤说明，使用了Markdown格式：

## 生成长度为4599的全零数组
运行 `shengcheng45990.py` 脚本，生成一个长度为4599，数据类型为 `int64` 的全零数组，并保存为 `int64.npy` 文件。

```bash
python shengcheng45990.py
```

## 数据维度转换
运行 `transform.py` 对训练和测试数据进行维度转换，将数据从 `(N, C, T, V, M)` 格式转换为 `(N, M, T, V, C)` 格式。

处理的文件包括：
- `train_joint.npy`
- `test_joint_B.npy`

```bash
python transform.py
```

## 生成训练集和测试集
1. 运行 `minger.py`，生成包含训练集的 `V1.npz` 文件，格式如下：
    ```
    V1.npz[[x_train], [y_train], [x_test], [y_test]]
    ```

    ```bash
    python minger.py
    ```

2. 运行 `mir.py`，生成包含测试集的 `V1.npz` 文件，格式如下：
    ```
    V1.npz[[[x_test], [y_test]]]
    ```

    ```bash
    python mir.py
    ```

## 转换为2D数据
如果模型需要2D数据：
1. 运行 `change2d.py`，将3D数据中的Z轴去除。
   
    ```bash
    python change2d.py
    ```

2. 运行 `transform2d.py`，对数据进行转换。

    ```bash
    python transform2d.py
    ```

3. 运行 `minger.py` 和 `mir.py` 生成2D版本的 `V1.npz` 文件。

    ```bash
    python minger.py
    python mir.py
    ```

## 文件路径和命名
将生成的文件拷贝到以下路径，并确保文件名与配置文件中的命名格式一致，默认命名为 `V1.npz`：
```
ICMEW2024-Track10/Model_inference/Mix_GCN/dataset/save_3d_pose
```