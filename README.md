# ICE-Pruning

The offical implmentations of the experiments of paper: <em><strong>ICE-Pruning: An Iterative Cost-Efficient Pruning Pipeline for Deep Neural Networks</em></strong>

The code that are used/modified in this repo for this paper are from:

https://github.com/jdg105/linearly-replaceable-filter-pruning<br>
https://github.com/kuangliu/pytorch-cifar<br>
https://github.com/pytorch/vision/tree/main/torchvision/models<br>
https://github.com/kaiqi123/Automatic-Attention-Pruning/tree/main<br>
https://github.com/tyui592/Pruning_filters_for_efficient_convnets/tree/master

The folders in this repo and experiments in the paper are one-to-one correspondence:

freezing_motivation/smaller_scale -- experiments in Section IV.B<br>
compare_pruning_criteria -- experiments in Section IV.C<br>
ablation_study -- experiments in Section IV.D<br>
SOTAs -- experiments in Section IV.E<br>

## Notes
We fixed the bug for multi-objective optimization in https://github.com/kaiqi123/Automatic-Attention-Pruning/tree/main

## Citation

```
@misc{hu2025icepruningiterativecostefficientpruning,
      title={ICE-Pruning: An Iterative Cost-Efficient Pruning Pipeline for Deep Neural Networks}, 
      author={Wenhao Hu and Paul Henderson and José Cano},
      year={2025},
      archivePrefix={arXiv},
}
```




