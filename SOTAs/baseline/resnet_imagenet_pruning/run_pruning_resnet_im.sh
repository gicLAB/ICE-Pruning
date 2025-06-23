for s in 0 
do
	for p in 0.8
	do
		python3 auto_tune_ice_pick_im.py --pruning_ratio $p --seed $s --pruning_method l1
	done
done
