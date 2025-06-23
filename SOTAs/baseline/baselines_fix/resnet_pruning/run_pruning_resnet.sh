for s in 0 
do
	for p in 0.6 0.7 0.8
	do
		python3 resnet_pruning/auto_tune_ice_pick.py --pruning_ratio $p --seed $s --pruning_method l1
	done
done
