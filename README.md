## Description
This is a implementation of mobilenet-ssd for face detection written by keras, which is the first step of my **FaceID system**. You can find another two repositories  as follows:
1. [Face-detection-with-mobilenet-ssd](https://github.com/bruceyang2012/Face-detection-with-mobilenet-ssd)
2. [Face-Alignment-with-simple-cnn](https://github.com/bruceyang2012/Face-Alignment-with-simple-cnn)
3. [Face-identification-with-cnn-triplet-loss](https://github.com/bruceyang2012/Face-identification-with-cnn-triplet-loss) 

## prepare data
1. You are advised to use CUDA-compatible GPUs to train the model.
2. Download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) from Official Website , and put it into data_path folder in [face_train.ipynb](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/face_train.ipynb).
3. [wider_extract.py](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/wider_extract.py) has been modified to show the method of exctracting faces from the datasets. It's easy to follow.

## train
1. Follow [face_train.ipynb](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/face_train.ipynb) step by step. You can change the parameters for better performance.
2. [wider_train_small.npy](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/wider_train_small.npy) and [wider_val_small.npy](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/wider_val_small.npy) are made to testing the network. If you don't have enough gpu resources, you can also use them for training.

## test
Here are some testing results. It seems good but improvement is still needed. For example, the Bbox is a little bit inaccurate.

<div align=center><img src="https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/38_Tennis_Tennis_38_683.jpg">    <img src="https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/2_Demonstration_Demonstration_Or_Protest_2_441.jpg"/></div>

<div align=center><img src="https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/10_People_Marching_People_Marching_2_307.jpg">    <img src="https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/12_Group_Group_12_Group_Group_12_331.jpg"/></div>

<div align=center><img src="https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/36_Football_americanfootball_ball_36_1021.jpg">    <img src="https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/54_Rescue_rescuepeople_54_1006.jpg"/></div>

## to do
1. Evaluation is on the way.
2. MobileNetV2 version.

## License
MIT LICENSE

## References
1. [rykov8/ssd_keras](https://github.com/rykov8/ssd_keras)
2. [rcmalli/keras-mobilenet](https://github.com/rcmalli/keras-mobilenet)
