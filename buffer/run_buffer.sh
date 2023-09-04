#buffer for FTD 
CUDA_VISIBLE_DEVICES=1 python buffer_FTD.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca \
--buffer_path=../buffer_storage/ --data_path=../dataset/ \
--rho_max=0.01 --rho_min=0.01 
