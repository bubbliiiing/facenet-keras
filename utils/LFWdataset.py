import numpy as np
import os
from PIL import Image

def letterbox_image(image, size):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class LFWDataset():
    def __init__(self, dir, pairs_path, image_size, batch_size):
        super(LFWDataset, self).__init__()
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.batch_size = batch_size
        self.validation_images = self.get_lfw_paths(dir)

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

    def generate(self):
        imgs1 = []
        imgs2 = []
        issames = []
        for annotation_line in self.validation_images:  
            (path_1, path_2, issame) = annotation_line
            img1, img2 = Image.open(path_1), Image.open(path_2)
            img1 = letterbox_image(img1,[self.image_size[1],self.image_size[0]])
            img2 = letterbox_image(img2,[self.image_size[1],self.image_size[0]])
            
            img1, img2 = np.array(img1)/255., np.array(img2)/255.

            imgs1.append(img1)
            imgs2.append(img2)
            issames.append(issame)
            if len(imgs1) == self.batch_size:
                imgs1 = np.array(imgs1)
                imgs2 = np.array(imgs2)
                issames = np.array(issames)
                yield imgs1, imgs2, issames
                imgs1 = []
                imgs2 = []
                issames = []

        imgs1 = np.array(imgs1)
        imgs2 = np.array(imgs2)
        issames = np.array(issames)
        yield imgs1, imgs2, issames