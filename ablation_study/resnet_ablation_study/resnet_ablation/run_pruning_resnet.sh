for s in 0 1 2
do
	for p in 0.6
	do
		python3 resnet_ablation/resnet_ice_ablation.py --pruning_ratio $p --seed $s --pruning_method l1 --no_threshold
		python3 resnet_ablation/resnet_ice_ablation.py --pruning_ratio $p --seed $s --pruning_method l1 --no_freeze
		python3 resnet_ablation/resnet_ice_ablation.py --pruning_ratio $p --seed $s --pruning_method l1 --no_adapt_lr
	done
done
