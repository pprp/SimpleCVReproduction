# Training for libfacedetection in PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

It is the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The source code is based on [FaceBoxes.PyTorch](https://github.com/sfzhang15/FaceBoxes.PyTorch) and [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).


### Contents
- [Installation](#installation)
- [Training](#training)
- [Detection](#detection)
- [Export CPP source code](#export-cpp-source-code)
- [Design your own model](#design-your-own-model)

## Installation
1. Install [PyTorch](https://pytorch.org/) >= v1.0.0 following official instruction.

2. Clone this repository. We will call the cloned directory as `$TRAIN_ROOT`.
```Shell
git clone https://github.com/ShiqiYu/libfacedetection.train
```

3. Install dependencies.
```shell
pip install -r requirements.txt
```

_Note: Codes are based on Python 3+._

## Training
1. Download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html) dataset, place the images under this directory:
  ```Shell
  $TRAIN_ROOT/data/WIDER_FACE_rect/images
  ```
  and create a symbol link to this directory from  
  ```Shell
  $TRAIN_ROOT/data/WIDER_FACE_landmark/images
  ```
2. Train the model using WIDER FACE:
  ```Shell
  cd $TRAIN_ROOT/tasks/task1/
  python3 train.py
  ```

## Detection
```Shell
cd $TRAIN_ROOT/tasks/task1/
./detect.py -m weights/yunet_final.pth --image_file=filename.jpg
```

## Evaluation on WIDER Face
1. Enter the directory.
```shell
cd $TRAIN_ROOT/tasks/task1/
```

2. Create a symbolic link to WIDER Face. `$WIDERFACE` is the path to WIDER Face dataset, which contains `wider_face_split/`, `WIDER_val`, etc.
```shell
ln -s $WIDERFACE widerface
```

3. Perform evaluation. To reproduce the following performance, run on the default settings. Run `python test.py --help` for more options.
```shell
mkdir results
python test.py
```

4. Download and run the [official evaluation tools](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip). ***NOTE***: Matlab required!
```shell
# download
wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
# extract
unzip eval_tools.zip
# run the offical evaluation script
cd eval_tools
vim wider_eval.m # modify line 10 and line 21 according to your case
matlab -nodesktop -nosplash -r "run wider_eval.m;quit;"
```

### Performance on WIDER Face (Val)
Run on default settings: scales=[1.], confidence_threshold=0.3:
```
AP_easy=0.849, AP_medium=0.816, AP_hard=0.601
```

## Export CPP source code
The following bash code can export a CPP file for project [libfacedetection](https://github.com/ShiqiYu/libfacedetection)
```Shell
cd $TRAIN_ROOT/tasks/task1/
./exportcpp.py -m weights/yunet_final.pth -o output.cpp
```
## Design your own model
You can copy $TRAIN_ROOT/tasks/task1/ to $TRAIN_ROOT/tasks/task2/ or other similar directory, and then modify the model defined in file: tasks/task2/yufacedetectnet.py .

