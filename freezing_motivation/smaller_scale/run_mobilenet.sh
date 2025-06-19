for s in 0 1 2
do
	python3 smaller_scale/trace_res.py --ratio_pruning 0.3 --model_name mobilenetv2 --seed $s
done
