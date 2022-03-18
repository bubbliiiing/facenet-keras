import math
import os
import random

import keras
import numpy as np
from keras.utils import np_utils
from PIL import Image

from .utils import cvtColor, preprocess_input, resize_image


#------------------------------------#
#   数据加载器
#------------------------------------#
class FacenetDataset(keras.utils.Sequence):
    def __init__(self, input_shape, lines, batch_size, num_classes, random):
        self.input_shape    = input_shape
        self.lines          = lines
        self.length         = len(lines)
        self.batch_size     = batch_size
        self.num_classes    = num_classes
        self.random         = random
        
        #------------------------------------#
        #   路径和标签
        #------------------------------------#
        self.paths  = []
        self.labels = []

        self.load_dataset()
        
    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        #------------------------------------#
        #   创建全为零的矩阵
        #------------------------------------#
        images = np.zeros((self.batch_size // 3, 3, self.input_shape[0], self.input_shape[1], 3))
        labels = np.zeros((self.batch_size // 3, 3))
        
        for i in range(self.batch_size // 3):
            #------------------------------------#
            #   先获得两张同一个人的人脸
            #   用来作为anchor和positive
            #------------------------------------#
            c               = random.randint(0, self.num_classes - 1)
            selected_path   = self.paths[self.labels[:] == c]
            while len(selected_path) < 2:
                c               = random.randint(0, self.num_classes - 1)
                selected_path   = self.paths[self.labels[:] == c]

            #------------------------------------#
            #   随机选择两张
            #------------------------------------#
            image_indexes = np.random.choice(range(0, len(selected_path)), 2)
            #------------------------------------#
            #   打开图片并放入矩阵
            #------------------------------------#
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))
            #------------------------------------------#
            #   翻转图像
            #------------------------------------------#
            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))

            images[i, 0, :, :, :] = image
            labels[i, 0] = c
            
            image = cvtColor(Image.open(selected_path[image_indexes[1]]))
            #------------------------------------------#
            #   翻转图像
            #------------------------------------------#
            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            
            images[i, 1, :, :, :] = image
            labels[i, 1] = c

            #------------------------------#
            #   取出另外一个人的人脸
            #------------------------------#
            different_c         = list(range(self.num_classes))
            different_c.pop(c)
            
            different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.paths[self.labels == current_c]
            while len(selected_path) < 1:
                different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
                current_c           = different_c[different_c_index[0]]
                selected_path       = self.paths[self.labels == current_c]

            #------------------------------#
            #   随机选择一张
            #------------------------------#
            image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
            image               = cvtColor(Image.open(selected_path[image_indexes[0]]))
            #------------------------------------------#
            #   翻转图像
            #------------------------------------------#
            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            
            images[i, 2, :, :, :] = image
            labels[i, 2] = current_c

        #--------------------------------------------------------------#
        #   假设batch为32
        #   0,32,64   属于一个组合 0和32是同一个人，0和64不是同一个人
        #--------------------------------------------------------------#
        images1 = np.array(images)[:, 0, :, :, :]
        images2 = np.array(images)[:, 1, :, :, :]
        images3 = np.array(images)[:, 2, :, :, :]
        images = np.concatenate([images1, images2, images3], 0)
        
        labels1 = np.array(labels)[:, 0]
        labels2 = np.array(labels)[:, 1]
        labels3 = np.array(labels)[:, 2]
        labels = np.concatenate([labels1, labels2, labels3], 0)

        labels = np_utils.to_categorical(np.array(labels), num_classes = self.num_classes)  
        
        return images, {'Embedding' : np.zeros_like(labels), 'Softmax' : labels}

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[0]))
        self.paths  = np.array(self.paths,dtype=np.object)
        self.labels = np.array(self.labels)

class LFWDataset():
    def __init__(self, dir, pairs_path, input_shape, batch_size):
        super(LFWDataset, self).__init__()
        self.input_shape        = input_shape
        self.pairs_path         = pairs_path
        self.batch_size         = batch_size
        self.validation_images  = self.get_lfw_paths(dir)

    def generate(self):
        images1 = []
        images2 = []
        issames = []
        for annotation_line in self.validation_images:  
            (path_1, path_2, issame)    = annotation_line
            image1, image2              = Image.open(path_1), Image.open(path_2)
            image1 = resize_image(image1, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image2 = resize_image(image2, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
                
            image1, image2 = preprocess_input(np.array(image1, np.float32)), preprocess_input(np.array(image2, np.float32))

            images1.append(image1)
            images2.append(image2)
            issames.append(issame)
            if len(images1) == self.batch_size:
                yield np.array(images1), np.array(images2), np.array(issames)
                images1     = []
                images2     = []
                issames     = []
                
        yield np.array(images1), np.array(images2), np.array(issames)

    def read_lfw_pairs(self,pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self,lfw_dir,file_ext="jpg"):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
        #for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list
