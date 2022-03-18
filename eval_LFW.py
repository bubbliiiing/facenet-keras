from nets.facenet import facenet
from utils.dataloader import LFWDataset
from utils.utils_metrics import test

if __name__ == "__main__":
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet、inception_resnetv1
    #--------------------------------------#
    backbone        = "mobilenet"
    #--------------------------------------#
    #   输入图像大小
    #--------------------------------------#
    input_shape     = [160, 160, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#
    model_path      = "model_data/facenet_mobilenet.h5"
    #--------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    #--------------------------------------#
    lfw_dir_path    = "lfw"
    lfw_pairs_path  = "model_data/lfw_pair.txt"
    #--------------------------------------#
    #   评估的批次大小和记录间隔
    #--------------------------------------#
    batch_size      = 256
    log_interval    = 1
    #--------------------------------------#
    #   ROC图的保存路径
    #--------------------------------------#
    png_save_path   = "model_data/roc_test.png"

    test_loader     = LFWDataset(dir=lfw_dir_path,pairs_path=lfw_pairs_path, batch_size=batch_size, input_shape=input_shape)

    model           = facenet(input_shape, backbone=backbone, mode="predict")
    model.load_weights(model_path, by_name = True)

    test(test_loader, model, png_save_path, log_interval, batch_size)