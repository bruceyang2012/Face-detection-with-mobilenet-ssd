## Description
This is a implemetation of mobile-ssd for face detection written by keras.

## prepare data
1. You need CUDA-compatible GPUs to train the model.
2. Download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) from Official Website , and put it into face_data folder in [face_generator.py](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/face_generator.py).

## train
Follow [face_train.ipynb](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/face_train.ipynb) step by step. You can change the parameters for better performance.

## test
Here are some testing results. It seems good but lots of improvement is needed.

![image](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/10_People_Marching_People_Marching_2_307.jpg)![image](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/12_Group_Group_12_Group_Group_12_331.jpg)
![image](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/36_Football_americanfootball_ball_36_1021.jpg)![image](https://github.com/bruceyang2012/face-detection-with-mobilenet-ssd/raw/master/output_test/54_Rescue_rescuepeople_54_1006.jpg)

## License
MIT LICENSE

## References
1. [face-detection](https://github.com/sudhakar-sah/face-detection)
2. [keras-mobilenet](https://github.com/rcmalli/keras-mobilenet)
