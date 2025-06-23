# ============================================ CIFAR-10, ResNet-56 ==================================================================
# The choices of prune_type are as follows:
# AAP_givenAccLoss_minimizePara
# AAP_givenAccLoss_minimizeFlops
# AAP_givenParaReduce
# AAP_givenFlopsReduce
# AAP_givenAccLossParaReduce

# Remember to change to following 2 lines in the model_module.py of the folder "pruning" as needed
# curr_conv_threshold = conv_threshold * threshold_weights_dict_para[name] # set threshold using weights calculated by parameters
# curr_conv_threshold = conv_threshold * threshold_weights_dict_flops[name] # set threshold using weights calculated by flops
# ====================================================================================================================================
# num_pruningRound 100 0.8 target_loss 0.035
# num_pruningRound 80 0.6 target_loss 0.035
# lamb 0.2 works
for p in 0.6 0.7 0.8
do
	for s in 0
	do
		CUDA_VISIBLE_DEVICES=0 python3 non_change/aap/main.py --data './cifar10' --num-classes 10 \
		--raport-file raport.json -j8 -p 100 \
		--workspace ${1:-./} --arch resnet152 -c fanin \
		--label-smoothing 0.0 --lr-schedule step --lr_decay_epochs '91,136' --warmup 0 --mom 0.9 --wd 0.0002 \
		--lr 0.001 --optimizer-batch-size 512 --batch-size 128 \
		--prune_type AAP_givenAccLossParaReduce_minimizePara \
		--epochs 182 --rewind_epoch 0 --target_accloss 0.035 --target_parameters_reduce $p --lambda_value 0.2 \
		--power_value 1 --seed $s
	done
done
