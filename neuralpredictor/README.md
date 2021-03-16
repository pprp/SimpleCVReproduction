# Neural Predictor for Neural Architecture Search

Wei Wen, Hanxiao Liu, Hai Li, Yiran Chen, Gabriel Bender, Pieter-Jan Kindermans. "Neural Predictor for Neural Architecture Search". arXiv:1912.00848. [Paper link](https://arxiv.org/abs/1912.00848).

**This is a open source reproduction in PyTorch.**

## Reproduction Results

![](assets/scatterplot.png)

All the results are run with the hyper-parameters provided in paper (default value in `train.py`), unless otherwise specified.

The following results are MSE. The lower, the better.

| Train Split | Eval Split | Paper | Reproduction | Comments                |
|-------------|------------|-------|--------------|-------------------------|
| 172         | all        | 1.95  | 3.62         |                         |
| 860         | all        | NA    | 2.94         |                         |
| 172         | denoise-80 | NA    | 1.90         |                         |
| 91-172      | denoise-91 | 0.66  | 0.74         | Paper used classifier to denoise |
| 91-172      | denoise-91 | NA    | 0.56         | epochs = 600, lr = 2e-4 |

NOTE: As the classifier is not ready, we cheated a little by directly filtering out all the architectures below 91%. The splits are called `91-`.

## TODO Items

- [ ] Classifier (first stage)
- [ ] Cross validation
- [ ] E2E Architecture Selection

## Preparation

### Dependencies

* PyTorch (cuda)
* [NasBench](https://github.com/google-research/nasbench/tree/master/nasbench)
* h5py
* matplotlib

### Dataset

Download HDF5 version of NasBench from [here](https://drive.google.com/open?id=1x1EQCyClzHBVDHloUCtES_-M_E9o4MeF) and put it under `data`.

Then generate train/eval split:

```
python tools/split_train_val.py
```

### Advanced: Build HDF5 Dataset from Scratch

Skip this step if you have downloaded the data from last step.

This step is to convert the tfrecord into a hdf5 file, as the official asset Google has provided is too slow to read (and very large in volume).

Download `nasbench_full.tfrecord` from NasBench, and put it under `data`. Then run

```
python tools/nasbench_tfrecord_converter.py
```

### Splits

The following splits are provided for now:

* `172`, `334`, `860`: Randomly sampled architectures from NasBench.
* `91-172`, `91-334`, `91-860`: The splits above filtered with a threshold (validation accuracy 91% on seed 0).
* `denoise-91`, `denoise-80`: All architectures filtered with threshold 91% and 80%.
* `all`.

## Train and Evaluation

Refer to `python train.py -h` for options. Training and evaluation are very fast (about 90 seconds on P100).

## Implementation Details

### HDF5 Format

The HDF5 is quite self-explanatory. You can refer to `dataset.py` for how to read it. The only thing I believe should be highlighted is that `metrics` is a 423624 x 4 (epochs: 4, 12, 36, 108) x 3 (seed: 0, 1, 2) x 2 (halfway, total) x 4 (`training_time`, `train_accuracy`, `validation_accuracy`, `test_accuracy`) matrix.

### Modeling and Training

* The paper didn't mention where to put dropout. So dropout is added after every layer. Nevertheless, the model still tends to overfit.
* Uses xavier uniform for initialization. Bias for linear layer is turned off.
* Paper didn't mention hyper-parameters other than lr and weight decay in Adam. We follow default settings in tensorflow 1.15.
* We drop the last samples (less than batch size) in every epoch.
* We normalize the labels (validation accuracy) with MEAN (90.8) and STD (2.4).
* Resample with different seed in case the validation accuracy belows (< 15%).

### Bad results for evaluation on "all"

A brief case study reveals that the bad results are mainly due to the "noise" in NasBench. In NasBench, there are two types of noises:

1. Some training (0.63%) "blows". The results are about ~10%. We resample on such case. We can handle 99.85% (422979/423624), with others using mean accuracy directly.
2. The result diversifies. About 1.2% of the architectures get accuracy lower than 80%, but they contribute a lot to MSE. We found that without sampling them in testing, the results improve and are on par with paper.

## References

- [graph-cnn.pytorch](https://github.com/meliketoy/graph-cnn.pytorch)
- [pygcn](https://github.com/tkipf/pygcn)
