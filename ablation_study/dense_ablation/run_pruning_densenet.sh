for s in 0 1 2
do
	for p in 0.6 0.7 0.8
	do
		python3 dense_ablation/densenet_ice_ablation.py --pruning_ratio $p --seed $s --pruning_method l1 --no_threshold
		python3 dense_ablation/densenet_ice_ablation.py --pruning_ratio $p --seed $s --pruning_method l1 --no_freeze
		python3 dense_ablation/densenet_ice_ablation.py --pruning_ratio $p --seed $s --pruning_method l1 --no_adapt_lr
	done
done
