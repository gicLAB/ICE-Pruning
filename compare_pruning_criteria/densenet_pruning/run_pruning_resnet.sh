for s in 0 1 2 
do
	for p in 0.6 0.7 0.8
	do
		python3 densenet_pruning/auto_tune_ice_pick_dense.py --pruning_ratio $p --seed $s --pruning_method l1
		python3 densenet_pruning/auto_tune_ice_pick_dense.py --pruning_ratio $p --seed $s --pruning_method random
		python3 densenet_pruning/auto_tune_ice_pick_dense.py --pruning_ratio $p --seed $s --pruning_method entropy
		python3 densenet_pruning/auto_tune_ice_pick_dense.py --pruning_ratio $p --seed $s --pruning_method mean_activation
	done
done
