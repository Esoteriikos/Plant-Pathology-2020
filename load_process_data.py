import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()


"""
    csv file : image id labels
  - image_id    target
    train_05    1
     
    images folder : all images
    train.05.jpg ..
"""

"""
    images_path : image folder path
    csv_path : csv file with image id path
    
"""
class data_from_id:
    def __init__(self, images_path, csv_path, img_shape=(150,150), target=False):
        try:
            self.img_shape = img_shape
            self.target = target
            self.image_label = None
            self.target_labels = None
            if os.path.exists(images_path) and os.path.exists(csv_path):
                self.images_path = images_path
                self.data = pd.read_csv(csv_path)
        except Exception as e:
            print(e)

    def load_image(self, image_id):
        filename = image_id + ".jpg"
        image = cv2.imread(self.images_path + filename)
        #image = image[220:1100, 350:1700]
        # cv2.imshow('frame', image)
        # cv2.waitKey(0)
        image = cv2.resize(image, self.img_shape)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    def load_target(self, target_id, tar):
        if self.target:
            targets = self.data.loc[target_id]#[self.target_labels])
            targets = targets[tar].sample(4)
            targets = np.array(targets).astype(np.float64).reshape(1, len(tar))
            print(targets)
            #print("impo", self.data[target_id][self.target_labels])
            #return targets
        else:
            print("TargetVariables NotFound:")
            return None

    def image_target_array(self, image_label, target_labels=[], random_state=-1):
        print(self.data.head())
        M = len(self.data.index)
        if random_state==-1:
            images_arr = self.data[image_label].progress_apply(self.load_image)
        else:
            images_arr = self.data[image_label].sample(M, random_state=random_state).progress_apply(self.load_image)
        w, h = self.img_shape
        images_arr = np.array([images_arr]).astype(np.int64).reshape(M, w, h, 3)
        
        if self.target and target_labels:
            self.target_labels = target_labels
            if random_state==-1:
                targets_arr = self.data[target_labels]
            else:
                targets_arr = self.data[target_labels].sample(M, random_state=random_state)
            targets_arr = np.array(targets_arr).astype(np.int64).reshape(M, len(target_labels))
            print(images_arr.shape, targets_arr.shape)
            return images_arr, targets_arr
        print(images_arr.shape)
        return images_arr


#set = data_from_id(r'plant-pathology-2020-fgvc7\images\\', r"plant-pathology-2020-fgvc7\train.csv", target=True)
#train, test = set.image_target_array(image_label='image_id',target_labels=['healthy', 'multiple_diseases', 'rust', 'scab'], random_state=-1)
#set.load_target(1502, ['healthy', 'multiple_diseases', 'rust', 'scab'])