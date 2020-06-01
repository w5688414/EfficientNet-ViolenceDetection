from keras import Input
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop,SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
from keras_layer_normalization import LayerNormalization
import sys
import logging
from keras.applications import Xception, ResNet50, InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from model.mobilenet_v3_large import MobileNetV3_Large
from model.mobilenet_v3_small import MobileNetV3_Small
from keras_efficientnets import EfficientNetB0
from keras_efficientnets import EfficientNetB1
from shufflenetv2 import ShuffleNetV2
from keras.layers import Permute
from keras.layers import multiply
from keras.layers import add
from keras.utils import plot_model


TIME_STEPS = 8
# first way attention
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    # a = Permute((2, 1))(inputs)
    a_probs = Dense(256, activation='softmax')(inputs)
    # a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    # output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    output_attention_mul = add([inputs, a_probs], name='attention_mul')
    return output_attention_mul



def build(size, seq_len , learning_rate ,
          optimizer_class ,\
          initial_weights ,\
          cnn_class ,\
          pre_weights , \
          lstm_conf , \
          cnn_train_type, classes = 1, dropout = 0.0):
    input_layer = Input(shape=(seq_len, size, size, 3))
    if(cnn_train_type!='train'):
        if cnn_class.__name__ == "ResNet50":
            cnn = cnn_class(weights=pre_weights, include_top=False,input_shape =(size, size, 3))
        elif cnn_class.__name__=="MobileNetV3_Large":
            cnn=cnn_class(shape =(size, size, 3),n_class=2,include_top=False).build()
        elif cnn_class.__name__=='MobileNetV3_Small':
             cnn=cnn_class(shape =(size, size, 3),n_class=2,include_top=False).build()
        elif cnn_class.__name__=='efn.EfficientNetB0':
            cnn = EfficientNetB0(input_shape=(size,size,3), classes=2, include_top=False, weights='imagenet')
        elif cnn_class.__name__=='efn.EfficientNetB1':
            cnn = EfficientNetB1(input_shape=(size,size,3), classes=2, include_top=False, weights='imagenet')
        elif cnn_class.__name__=="ShuffleNetV2":
            cnn=ShuffleNetV2(include_top=False,input_shape=(224, 224, 3),bottleneck_ratio=1)
        else:
            cnn = cnn_class(weights=pre_weights,include_top=False)
    else:
        cnn = cnn_class(include_top=False)

    #control Train_able of CNNN
    if(cnn_train_type=='static'):
        for layer in cnn.layers:
            layer.trainable = False
    if(cnn_train_type=='retrain'):
        for layer in cnn.layers:
            layer.trainable = True

    cnn = TimeDistributed(cnn)(input_layer)
    print(cnn)
    #the resnet output shape is 1,1,20148 and need to be reshape for the ConvLSTM filters
    # if cnn_class.__name__ == "ResNet50":
        # cnn = Reshape((seq_len,4, 4, 128), input_shape=(seq_len,1, 1, 2048))(cnn)
    # print(lstm_conf)
    # print(lstm_conf[0])
    # print(lstm_conf[1])
    lstm = lstm_conf[0](**lstm_conf[1])(cnn)
    lstm = MaxPooling2D(pool_size=(2, 2))(lstm)
    attention_mul = attention_3d_block(lstm)
    # print(lstm)
    # lstm = MaxPooling2D(pool_size=(2, 2))(lstm)
    flat = Flatten()(attention_mul)

    flat = BatchNormalization()(flat)
    # flag=LayerNormalization()(flat)
    flat = Dropout(dropout)(flat)
    linear = Dense(512)(flat)

    relu = Activation('relu')(linear)
    linear = Dense(256)(relu)
    linear = Dropout(dropout)(linear)
    relu = Activation('relu')(linear)
    linear = Dense(10)(relu)
    linear = Dropout(dropout)(linear)
    relu = Activation('relu')(linear)

    activation = 'sigmoid'
    loss_func = 'binary_crossentropy'

    if classes > 1:
        activation = 'softmax'
        loss_func = 'categorical_crossentropy'
    predictions = Dense(classes,  activation=activation)(relu)

    model = Model(inputs=input_layer, outputs=predictions)
    optimizer = optimizer_class[0](lr=learning_rate, **optimizer_class[1])
    model.compile(optimizer=optimizer, loss=loss_func,metrics=['acc'])

    print(model.summary())
    plot_model(model,show_shapes=True,to_file="model.png")


    return model