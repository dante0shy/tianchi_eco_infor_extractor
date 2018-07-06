import glob
import json
import random
import math
import numpy as np
import cv2
class Data_loader():
    def __init__(self,input,char_list, label_list ,word_len = 400):
        self.word_len = word_len
        self.input_file = json.load(open(input,'r'))
        # self.input_file =[f for f in self.input_file if '/56/' in f]
        self.label_list = json.load(open(label_list,'r'))
        self.char_list = json.load(open(char_list,'r'))
        self.data = self._build_data_lsit(self.input_file)

        # self.batch_size =batch_size
        pass

    def _build_data_lsit(self,input_file):
        detail_list = []
        label_list = []
        mask_list = []
        for input in input_file:
            data = json.load(open(input,'r'))
            angle = self.get_rotation_degree(data)
            if abs(angle) >=0. and abs(angle) <=10.:
                for idx,info in enumerate(data):
                    data[idx]['location'] = map(lambda x: self.coord_rotation(-angle, x), info['location'])
            tmp = [data[0]]
            tmp.extend(sorted(data[1:], key=lambda x: x['location'][0][1]))
            data = tmp
            scale = []
            detail_one = []
            label_one = []
            mask_one = []
            mask_list.append(len(data))
            for idx in range(self.word_len+1):
                if idx>=len(data):
                    detail_one.append([0]*(len(self.char_list)+9+30))
                    label_one.append(0)
                    continue
                if not idx:
                    scale= reduce(self.compare_normal, data[idx]['location'])
                else:
                    tmp = [0]*len(self.char_list)
                    chars = map(
                        lambda x : self.char_list.index(x)
                            if x<=u'\u007f'
                            else self.char_list.index('#unicode') ,
                        list(data[idx]['text'])
                                    )
                    for c in chars:
                        tmp[c] +=1
                    detail = []
                    for x,y in data[idx]['location']:
                        # detail.extend([float(x)/scale[0],float(y)/scale[1]])
                        detail.extend([float(x)/100.,float(y)/100.])

                    brand =31*[0]
                    brand[data[idx]['label_brand']]=1
                    detail.extend(brand)

                    tmp.extend(detail)
                    try:
                        if data[idx]['label'] == 'pt' or   data[idx]['label'] == 'po':
                            data[idx]['label'] = 'p'
                        try:
                            if '.' in data[idx]['text'] or ',' in data[idx]['text']:
                                float(data[idx]['text'].replace(',','.'))
                                data[idx]['label'] = 'p'
                        except:
                            pass
                        try:
                            if data[idx]['label'] == 'd' and data[idx]['label_brand']== 3 and int(data[idx]['text']):
                                data[idx]['label'] = 'o'
                        except:
                            pass

                        # if data[idx]['label'] == 'loc':
                        #     data[idx]['label'] = 'o'
                        # if data[idx]['label'] not in  ['pc','dt','p','d','t']:
                        #     data[idx]['label'] = 'o'

                        label = self.label_list.index(data[idx]['label'])
                    except:
                        label = self.label_list.index('o')
                    detail_one.append(tmp)
                    # tmp_label = [0]*len((self.label_list))
                    # tmp_label[label] += 1
                    label_one.append(label)
                pass
            detail_list.append(detail_one)
            label_list.append(label_one)
            # mask_list.append(mask_one)
        return detail_list,label_list,mask_list
        pass

    def load_data_batch(self,batch_size):
        start = 0
        # random.shuffle(self.data)
        for _ in range(len(self.input_file)/batch_size+1):
            if start+batch_size <len(self.input_file):
                end = start+batch_size
                yield self.data[0][start:end],self.data[1][start:end],self.data[2][start:end]
                start = end
            else:
                yield self.data[0][start:],self.data[1][start:],self.data[2][start:]

    def coord_rotation(self, angle, pts):
        cos_v = math.cos(math.radians(angle))
        sin_v = math.sin(math.radians(angle))
        pts_x_new = pts[0] * cos_v - pts[1] * sin_v
        pts_y_new = pts[0] * sin_v + pts[1] * cos_v

        return [pts_x_new,pts_y_new]#{'y': pts_y_new, 'x': pts_x_new}

    def get_rotation_degree(self, text_response):
        angles = []
        for idx, each_response in enumerate(text_response):  # @UnusedVariable
            if idx == 0:
                continue
            x = [0, 0, 0, 0]
            y = [0, 0, 0, 0]
            for i in range(0, 4):
                x[i]=each_response['location'][i][0] # x response 4 points
                y[i]=each_response['location'][i][1] # x response 4 points
            angle1 = cv2.fastAtan2(abs(y[1] - y[0]), x[1] - x[0])
            if y[1] < y[0]:
                angle1 *= -1
            angle2 = cv2.fastAtan2(abs(y[2] - y[3]), x[2] - x[3])
            if y[2] < y[3]:
                angle2 *= -1

            angles.extend([angle1, angle2])
        return np.median(angles)
    def rotateAndScale(self,img, scaleFactor=1, degreesCCW=270):
        oldY, oldX = img.shape[:2]  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
        M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                    scale=scaleFactor)  # rotate about center of image.
        newX, newY = oldX * scaleFactor, oldY * scaleFactor
        r = np.deg2rad(degreesCCW)
        newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))
        (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
        M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
        M[1, 2] += ty
        rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
        return rotatedImg
    @staticmethod
    def compare_normal(x, y):
        x_m = max(x[0], y[0])
        y_m = max(x[1], y[1])
        return  x_m,  y_m


if __name__=='__main__':
    d = Data_loader('/home/xinmatrix/TwoTB/PycharmProjects/deja_receipt_ocr_v3/lstm_crf/data/train.json',
                    '/home/xinmatrix/TwoTB/PycharmProjects/deja_receipt_ocr_v3/lstm_crf/data/char_list_v2.json',
                    '/home/xinmatrix/TwoTB/PycharmProjects/deja_receipt_ocr_v3/lstm_crf/data/label.json'
                    )
    for asdas in d.load_data_batch(1):
        print(len(asdas[0][0][0]))
    pass