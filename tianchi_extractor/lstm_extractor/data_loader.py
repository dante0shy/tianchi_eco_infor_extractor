# -*- coding: UTF-8 -*-

import glob
import json
import random
import math
import numpy as np
import cv2
import os
from tianchi_extractor.lstm_extractor.tf_config import Config,path_format
from tianchi_extractor.config import TASKS

class Data_loader():
    def __init__(self,input,word_len = 400):
        self.word_len = word_len
        self.input_file = json.load(open(input,'r'))
        # self.input_file =[f for f in self.input_file if '/56/' in f]
        # self.label_list = json.load(open(label_list,'r'))
        # self.char_list = json.load(open(word_list,'r'))
        self.data = self._build_data_lsit(self.input_file)

        # self.batch_size =batch_size
        pass

    def _build_data_lsit(self,input_file):
        detail_list = []
        sent_list = []
        label_list = []
        mask_list = []
        max_len = 0
        for idx,input in enumerate(input_file):
            path = path_format.format(input[0],input[1])
            task = TASKS.index(input[0])
            tmp = [0]*3
            tmp [task]=1
            task = tmp
            if not os.path.exists(path):
                continue
            data = json.load(open(path,'r'))
            split_pos = filter(lambda x :x[1][0]==u'。',enumerate(data))
            data_split = []
            start = 0
            for idx,pos in enumerate(split_pos):
                if len(data[start:]) > max_len:
                    max_len = len(data[start:pos[0]+1])
                data_split.append(data[start:pos[0]+1])
                start = pos[0]+1
            if start<len(data):
                if len(data[start:]) > max_len:
                    max_len = len(data[start:])
                data_split.append(data[start:])

            for d in data_split:
                data_one = np.array(d)
                mask_one = len(data_one[:, 3].tolist())
                task_one  = [task]*len(mask_one)
                word_one = data_one[:,2].tolist()
                label_one = map(lambda x:list(x)[int(0.5*len(x))],data_one[:,3].tolist())
                if len(word_one)<Config.max_vector_len:
                    word_one.extend([Config.word_len-1]*(Config.max_vector_len-len(word_one)))
                    label_one.extend([]*(Config.max_vector_len-len(word_one)))
                    task_one.extend([task]*(Config.max_vector_len-len(word_one)))
                else:
                    word_one = word_one[:Config.max_vector_len]
                    label_one = label_one[:Config.max_vector_len]
                    task_one = task_one[:Config.max_vector_len]
                pass
                sent_list.append(word_one)
                detail_list.append(task_one)
                label_list.append(label_one)
                mask_list.append(mask_one)
            pass
        print(max_len)

        return sent_list, detail_list,label_list,mask_list
        pass

    def load_data_batch(self,batch_size):
        start = 0
        new_order = range(len(self.data[0]))
        random.shuffle(new_order)
        for _ in range(len(self.data[0]) / batch_size + 1):
            if start + batch_size < len(self.input_file):
                end = start + batch_size
                data_out = new_order[start:end]
                start = end
            else:
                data_out = new_order[start:]
            if not new_order:
                break
            yield [self.data[0][d] for d in data_out], \
                      [self.data[1][d] for d in data_out], \
                      [self.data[2][d] for d in data_out],\
                      [self.data[3][d] for d in data_out],

if __name__=='__main__':
    d = Data_loader('/home/xinmatrix/TwoTB/tianchi/extras/train.json',)
    for asdas in d.load_data_batch(1):
        print(len(asdas[0][0][0]))
    pass