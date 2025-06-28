# ICE-Pruning

The offical implmentations of the experiments of paper: <em><strong>ICE-Pruning: An Iterative Cost-Efficient Pruning Pipeline for Deep Neural Networks</em></strong>

The following table shows the links to the code that are used/modified in this repo for the paper:

|Code Repo Link|Used/Modified for|
|--------------|-----------------|
|https://github.com/jdg105/linearly-replaceable-filter-pruning|ResNet-152 model|
|https://github.com/kuangliu/pytorch-cifar |MobileNetV2 model|
|https://github.com/pytorch/vision/tree/main/torchvision/models|other models in this paper|
|https://github.com/kaiqi123/Automatic-Attention-Pruning/tree/main|AAP|
|https://github.com/tyui592/Pruning_filters_for_efficient_convnets/tree/master|L1 norm filter pruning and other pruning underlying code|

The folders in this repo and experiments in the paper are one-to-one correspondence:

|Folder|Experiments|
|------|-----------|
|freezing_motivation/smaller_scale|experiments in Section IV.B|
|compare_pruning_criteria|experiments in Section IV.C|
|ablation_study|experiments in Section IV.D|
|SOTAs|experiments in Section IV.E|

## Notes
1.We fixed the bug for multi-objective optimization in https://github.com/kaiqi123/Automatic-Attention-Pruning/tree/main<br>
2.The folders or files that have names as *_im are for ImageNet related experiments.
## Citation

```
@misc{hu2025icepruningiterativecostefficientpruning,
      title={ICE-Pruning: An Iterative Cost-Efficient Pruning Pipeline for Deep Neural Networks}, 
      author={Wenhao Hu and Paul Henderson and José Cano},
      year={2025},
      archivePrefix={arXiv},
}
```




