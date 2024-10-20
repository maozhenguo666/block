# block

# ctrgcn改blockgcn

# 环境安装
安装mix_GCN.yml,然后是
cd ./Model_inference/Mix_GCN
```
pip install -e torchlight
```
再装
```
pip install torch_topological
```


# Model inference
## Run Block_gcn
先数据预处理，在"Process_data\Process"的readme里
 
 ## Train

```

python main.py --config ./config /ctrgcn_V1_J_3d.yaml --device 0
python main.py --config ./config/ctrgcn_V1_B_3d.yaml --device 0
python main.py --config ./config/ctrgcn_V1_JM_3d.yaml --device 0
python main.py --config ./config/ctrgcn_V1_BM_3d.yaml --device 0

```
## Eval

```
python main.py --config ./config/ctrgcn_V1_J_3d.yaml --phase test --save-score True --weights best.pt --device 0
python main.py --config ./config/ctrgcn_V1_B_3d.yaml --phase test --save-score True --weights best.pt --device 0
python main.py --config ./config/ctrgcn_V1_JM_3d.yaml --phase test --save-score True --weights best.pt --device 0
python main.py --config ./config/ctrgcn_V1_BM_3d.yaml --phase test --save-score True --weights best.pt --device 0
```

## Ensemble
```
python Ensemble.py
```

