# FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search


## Introduction

One of the most critical problems in two-stage weight-sharing neural architecture search is the evaluation of candidate models. A faithful ranking certainly leads to accurate searching results. However, current methods are prone to making misjudgments. In this paper, we prove that they inevitably give biased evaluations due to inherent unfairness in the supernet training. In view of this, we propose two levels of constraints: expectation fairness and strict fairness. Particularly, strict fairness ensures equal optimization opportunities for all choice blocks throughout the training, which neither overestimates nor underestimates their capacity. We demonstrate this is crucial to improving confidence in models’ ranking (See Figure 1). Incorporating our supernet trained under fairness constraints with a multi-objective evolutionary search algorithm, we obtain various state-of-the-art models on ImageNet. Especially, FairNAS-A attains 77.5% top-1 accuracy.

![](images/fairnas-fig-1.png)
*Figure 1: Supernet Ranking Ability & Cost*

![](images/fairnas-sampling.png)
*Figure 2: FairNAS Supernet Training*

![](images/fairnas-architectures.png)
*FairNAS-A,B,C Architectures*

## Requirements
* Python 3.6 +
* Pytorch 1.0.1 +


## Updates
* Jul-3-2019： Model release of FairNAS-A, FairNAS-B, FairNAS-C.
* May-19-2020：Model release of FairNAS-A-SE, FairNAS-B-SE, FairNAS-C-SE and transfered models on CIFAR-10.

## Performance Result
![](images/result.png)
![](images/fairnas-cifar10.png)
![](images/fairnas-coco.png)

## Preprocessing
We have reorganized all validation images of the ILSVRC2012 ImageNet by their classes.

1. Download ILSVRC2012 ImageNet dataset.

2. Change to ILSVRC2012 directory and run the preprocessing script with
    ```
     ./preprocess_val_dataset.sh
    ```

## Evaluate

To evaluate,
    
    python3 verify.py --model [FairNAS_A|FairNAS_B|FairNAS_C] --device [cuda|cpu] --val-dataset-root [ILSVRC2012 root path] --pretrained-path [pretrained model path]

## Validate Transferred Model Accuracy

```
python transfer_verify.py --model [fairnas_a|fairnas_b|fairnas_c] --model-path pretrained/fairnas_[a|b|c]_transfer.pt.tar --gpu_id 0 --se-ratio 1.0 
```

Results:


    FairNAS-A-SE-1.0: flops: 403.36264M, params: 5.835322M, top1: 98.3, top5: 99.99
    FairNAS-B-SE-1.0: flops: 370.921184M, params: 5.603242M top1: 98.08, top5: 99.99
    FairNAS-C-SE-1.0: flops: 345.228096M, params: 5.42953M  top1: 98.01, top5: 99.99
    FairNAS-A-SE-0.5: flops: 414.305856M, params: 4.61373M top1: 98.15, top5: 99.98
    FairNAS-B-SE-0.5: flops: 358.330632M, params: 4.42485M, top1: 98.15, top5: 99.99
    FairNAS-C-SE-0.5: flops: 333.272088M, params: 4.283586M, top1: 97.99, top5: 99.99


## Validate FairNAS-SE models

```
python verify_se.py --val-dataset-root [ILSVRC2012 root path] --device cuda --model [fairnas_a|fairnas_b|fairnas_c] --model-path pretrained/fairnas_[a|b|c]_se.pth.tar 
```

Results:

    FairNAS-A-SE: mTop1: 77.5480	mTop5: 93.674000
    FairNAS-B-SE: mTop1: 77.1900	mTop5: 93.494000
    FairNAS-C-SE: mTop1: 76.6700	mTop5: 93.258000
    FairNAS-A-SE-0.5: mTop1: 77.3960	mTop5: 93.650000
    FairNAS-B-SE-0.5: mTop1: 77.1060	mTop5: 93.528000
    FairNAS-C-SE-0.5: mTop1: 76.7600	mTop5: 93.318000


## Citation

Your kind citations are welcomed!

    @article{chu2019fairnas,
        title={FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search},
        author={Chu, Xiangxiang and Zhang, Bo and Xu, Ruijun},
        journal={arXiv preprint arXiv:1907.01845},
        url={https://arxiv.org/pdf/1907.01845.pdf},
        year={2019}
    }
