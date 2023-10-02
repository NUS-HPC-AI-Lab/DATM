# Towards Lossless Dataset Distillation via Difficulty-Aligned Trajectory Matching

## [Project Page]() | [Paper]() | [Distilled Datasets]()
In this work, we find the difficulty of the generated patterns should be aligned with the size of the synthetic dataset.

What do easy patterns and hard patterns look like?

![image](figures/visualization.png)

After the alignment, dataset distillation can keep effective in both low- and high- IPC cases.

![image](figures/visualization_ipc.png)
## Getting Started
1. Create environment
```
conda env create -f environment.yaml
conda activate distillation
```
2. Generate expert trajectories
```
cd buffer
python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --train_epochs=100 --num_experts=100 --zca --buffer_path=../buffer_storage/ --data_path=../dataset/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --lr_teacher=0.01 --mom=0. --batch_train=256
```
3. Perform the distillation
```
cd distill
python DATM.py --cfg ../configs/xxxx.yaml
```
