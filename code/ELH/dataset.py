""" train and test dataset

author baiyu
"""
import copy
import os
import sys
import pickle
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import ImageFile
import math
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FITrain(Dataset):
    
    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        labels=os.listdir(os.path.join(path, 'train'))
        self.path=[]
        self.label={}
        for label in labels:
            tpath=os.path.join(path, 'train',label)
            imgs = os.listdir(tpath)
            
            for img in imgs:
                self.path.append(os.path.join(tpath,img))
                if label=='Amusement':
                    rlabel=0
                elif label=='Contentment':
                    rlabel=1
                elif label=='Awe':
                    rlabel=2 
                elif label=='Excitement':
                    rlabel=3
                elif label=='Fear':
                    rlabel=4
                elif label=='Sadness':
                    rlabel=5
                elif label=='Disgust':
                    rlabel=6
                elif label=='Anger':
                    rlabel=7
                else:
                    raise SystemExit('it failed!')
                    
                self.label[os.path.join(tpath,img)]=rlabel
        
        self.transform=transform
        
        
    def __len__(self):
        return len(self.path)       
    
    def __getitem__(self, index):
        image = Image.open(self.path[index]).convert('RGB')
        label=self.label[self.path[index]]
        if self.transform:
            image = self.transform(image)
        
        return image,label
    
class FItest(Dataset): 
    
    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
       
        labels=os.listdir(os.path.join(path, 'test'))
        self.path=[]
        self.label={}
        for label in labels:
            tpath=os.path.join(path, 'test',label)
            imgs = os.listdir(tpath)
            
            for img in imgs:
                self.path.append(os.path.join(tpath,img))
                if label=='Amusement':
                    rlabel=0
                elif label=='Contentment':
                    rlabel=1
                elif label=='Awe':
                    rlabel=2 
                elif label=='Excitement':
                    rlabel=3
                elif label=='Fear':
                    rlabel=4
                elif label=='Sadness':
                    rlabel=5
                elif label=='Disgust':
                    rlabel=6
                elif label=='Anger':
                    rlabel=7
                else:
                    raise SystemExit('it failed!')
                self.label[os.path.join(tpath,img)]=rlabel
        self.transform=transform
    def __len__(self):
        return len(self.path)       
    
    def __getitem__(self, index):
        image = Image.open(self.path[index]).convert('RGB')
        label=self.label[self.path[index]]
        if self.transform:
            image = self.transform(image)
        return  image,label


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json

from PIL import Image


class EmoSet(Dataset):
    ATTRIBUTES_MULTI_CLASS = [
        'scene', 'facial_expression', 'human_action', 'brightness', 'colorfulness',
    ]
    ATTRIBUTES_MULTI_LABEL = [
        'object'
    ]
    NUM_CLASSES = {
        'brightness': 11,
        'colorfulness': 11,
        'scene': 254,
        'object': 409,
        'facial_expression': 6,
        'human_action': 264,
    }

    def __init__(self,
                 args,
                 num_emotion_classes,
                 phase,
                 noise_type,
                 ):
        assert num_emotion_classes in (8, 2)
        assert phase in ('train', 'val', 'test')
        self.transforms_dict = self.get_data_transforms()
        data_root=args.path
        self.info = self.get_info(data_root, num_emotion_classes)
        self.phase=phase
        if phase == 'train':
            self.transform = self.transforms_dict['train']
        elif phase == 'val':
            self.transform = self.transforms_dict['val']
        elif phase == 'test':
            self.transform = self.transforms_dict['test']
        else:
            raise NotImplementedError

        data_store = json.load(open(os.path.join(data_root, f'{phase}.json')))
        self.data_store = [
            [
                self.info['emotion']['label2idx'][item[0]],
                item[0],
                os.path.join(data_root, item[1]),
                os.path.join(data_root, item[2])
            ]
            for item in data_store
        ]
      
    @classmethod
    def get_data_transforms(cls):
        transforms_dict = {

            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
            ]),

            'val': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ]),
            'test': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ]),
        }
        return transforms_dict

    def get_info(self, data_root, num_emotion_classes):
        assert num_emotion_classes in (8, 2)
        info = json.load(open(os.path.join(data_root, 'info.json')))
        if num_emotion_classes == 8:
            pass
        elif num_emotion_classes == 2:
            emotion_info = {
                'label2idx': {
                    'amusement': 0,
                    'awe': 0,
                    'contentment': 0,
                    'excitement': 0,
                    'anger': 1,
                    'disgust': 1,
                    'fear': 1,
                    'sadness': 1,
                },
                'idx2label': {
                    '0': 'positive',
                    '1': 'negative',
                }
            }
            info['emotion'] = emotion_info
        else:
            raise NotImplementedError

        return info

    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image

    def load_annotation_by_path(self, path):
        json_data = json.load(open(path))
        return json_data

    def __getitem__(self, item):
        emotion_label_idx, image_id, image_path, annotation_path = self.data_store[item]
        image = self.load_image_by_path(image_path)
        
        if self.phase == 'train':
            
            return image, emotion_label_idx
        if self.phase == 'test':
            return image, emotion_label_idx

    def __len__(self):
        return len(self.data_store)