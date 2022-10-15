# houseNet
This is a project I did during my Master program.

## Overview
This neural network project was created to evaluate the effect of dilated convolutions on a semantic segmentation task.
The dataset from the [crowdAI Mapping Challenge](https://www.crowdai.org/challenges/mapping-challenge)
will be used to test dilated convolutions for binary segmentation.
Check out the [wiki](https://github.com/mcFloskel/houseNet/wiki) for further explanations. 

## Usage
The neural network for this project was programmed by using [Keras](https://keras.io/) with the [Tensorflow](https://www.tensorflow.org/) backend.
Loading and visualizing image data is done with [OpenCV](https://opencv.org/).
The scripts should be usable if you have installed the above mentioned packages.
Keep in mind that this project was designed with the GPU version of Tensorflow and training will be slow or may not work at all with your CPU.

The train.py file can be used to start training if you have the mapping-challenge dataset available and created a configuration file.

## Dataset
A description of the dataset can be found [here](https://github.com/crowdAI/mapping-challenge-starter-kit).
The intended training/tests will be conducted with down-sampled images with the shape (150, 150, 3).
The dataset is loaded sequentially during training by using the [DataLoader](https://github.com/mcFloskel/houseNet/blob/master/util/data_loader.py).
The images/labels are loaded in batches and are randomly flipped or rotated.

Previous training/testing was done on down-sampled versions of the images from the mapping challenge.
The images were converted to [NumPy](http://www.numpy.org/) arrays with the shape (150, 150, 3) and stored in single .npy files.
If you convert the data and labels to numpy arrays you can use the [NumpyDataLoader](https://github.com/mcFloskel/houseNet/blob/master/util/data_loader.py) for training.


## Configuration
For loading the training/validation data a .ini file is read.
The file should contain the following content:

```ini
[DIRECTORIES]
train_data  = /path/to/data/train/
val_data    = /path/to/data/val/
models      = /path/to/models/
logs        = /path/to/logs/
predictions = /path/to/predictions/
```

## Current state
The project is finished and the evaluation can be found [here](https://github.com/mcFloskel/houseNet/wiki/Evaluation).
It is currently not planned to test further architectures in the near future.
However I am open for any suggestions for improvements.

## Example
Below you can see an example prediction of the [UNet](https://github.com/mcFloskel/houseNet/blob/master/networks/uNet3.py) architecture.
(Left to right: original image, ground truth, prediction)

<p align="center">
  <img src="https://github.com/mcFloskel/houseNet/blob/master/images/predictions/prediction_uNet.png"/>
</p>