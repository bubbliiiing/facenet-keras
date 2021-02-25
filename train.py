import os

import keras
import numpy as np
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import SGD, Adam

from nets.facenet import facenet
from nets.facenet_training import FacenetDataset, LFW_callback, triplet_loss


def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

if __name__ == "__main__":
    log_dir = "./logs/"
    annotation_path = "./cls_train.txt"
    num_classes = get_num_classes(annotation_path)
    #--------------------------------------#
    #   输入图像大小
    #--------------------------------------#
    # input_shape = [112,112,3]
    input_shape = [160,160,3]
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------#
    backbone = "mobilenet"

    model = facenet(input_shape, num_classes, backbone=backbone, mode="train")
    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    model_path = "model_data/facenet_mobilenet.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)

    #----------------------#
    #   LFW估计
    #----------------------#
    lfw_callback = LFW_callback("./lfw", input_shape)

    #-------------------------------------------------------#
    #   0.05用于验证，0.95用于训练
    #-------------------------------------------------------#
    val_split = 0.05
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    if backbone=="mobilenet":
        freeze_layer = 81
    elif backbone=="inception_resnetv1":
        freeze_layer = 440
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))

    for i in range(freeze_layer):
        model.layers[i].trainable = False
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        Batch_size = 64
        Lr = 1e-3
        Init_epoch = 0
        Freeze_epoch = 50
        
        model.compile(
            loss={
                'Embedding' : triplet_loss(batch_size=Batch_size),
                'Softmax'   : "categorical_crossentropy",
            }, 
            optimizer = Adam(lr=Lr),
            metrics={
                'Softmax'   : 'accuracy'
            }
        )
        print('Train with batch size {}.'.format(Batch_size))

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes, Batch_size)
        val_dataset   = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes, Batch_size)
            
        model.fit_generator(train_dataset,
                steps_per_epoch=max(1,num_train//Batch_size),
                validation_data=val_dataset,
                validation_steps=max(1, num_val//Batch_size),
                epochs=Freeze_epoch,
                initial_epoch=Init_epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, lfw_callback])

    for i in range(freeze_layer):
        model.layers[i].trainable = True
    if True:
        Batch_size = 32
        Lr = 1e-4
        Freeze_epoch = 50
        Epoch = 100
        
        model.compile(
            loss={
                'Embedding'     : triplet_loss(batch_size=Batch_size),
                'Softmax'       : "categorical_crossentropy",
            }, 
            optimizer = Adam(lr=Lr),
            metrics={
                'Softmax'       : 'accuracy'
            }
        )
        print('Train with batch size {}.'.format(Batch_size))

        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes, Batch_size)
        val_dataset   = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes, Batch_size)
            
        model.fit_generator(train_dataset,
                steps_per_epoch=max(1,num_train//Batch_size),
                validation_data=val_dataset,
                validation_steps=max(1, num_val//Batch_size),
                epochs=Epoch,
                initial_epoch=Freeze_epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, lfw_callback])
