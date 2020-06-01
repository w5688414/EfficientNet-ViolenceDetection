# Learning to Detect Violent Videos using Convolution LSTM

+ This work is based on violence detection model proposed by [1] with minor modications.
+ The original model was implemented with Pytorch [2] while in this work we implement it with Keras and TensorFlow as a back-end. 
+ The model incorporates pre-trained convolution Neural Network (CNN) connected to Convolutional LSTM (ConvLSTM) layer.
+ The model takes as an inputs the raw video, converts it into frames and output a binary classication of violence or non-violence label.
+ 主体部分主要来自于github的一个项目，运行也挺顺利的，所以我也就把握改的代码给开源出来，供其他人来参考
+ 尝试了很多主流的分类模型来替换ResNet部分，发现EfficientNet表现得很稳定，参数量还比残差等网络小很多，比现有的MobileNet,ShuffleNet这些精度更高，所以是一个折中的选择。
+ 加入了注意力机制，是一个小小的尝试，发现效果更好一点，看来是加对了，哈哈哈
+ 还加了很多传统方法的对比实验，有兴趣的人可以研究一下

### Architecture structure
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Architecture.jpeg)


## Running configurations
### Video datasets paths:
data path are defined as follows:
- hocky - data/raw_videos/HockeyFights
- violentflow - data/raw_videos/violentflow
- movies - data/raw_videos/movies

### Libraries perquisites:
- python 3.x
- numpy 1.14.0
- keras 2.2.0
- tensorflow 1.9.0
- Pillow 3.1.2
- opencv-python 3.4.1.15
- keras_efficientnets

### Running operation:
just run python run.py
(currently we don't support arguments from command line)

## Results
#### Hyper-tuning results
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/hyperparameters_results.JPG)

#### Hockey dataset results
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Hockey_results.png)

## Refrences
1. Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to detect violent videos
using convolution long short-term memory." In Advanced Video and Signal Based
Surveillance (AVSS), 2017 14th IEEE International Conference on, pp. 1-6. IEEE, 2017.
2. https://github.com/swathikirans/violence-recognition-pytorch
