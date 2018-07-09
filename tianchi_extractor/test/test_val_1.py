from tianchi_extractor.lstm_extractor.lstm_crf_classifier import Classifier
from tianchi_extractor.lstm_extractor.text_process import Data_loader
import json
import glob

import os
if __name__=='__main__':
    out_dir = '/home/xinmatrix/TwoTB/tianchi/extras/test_res'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # input =  '/home/xinmatrix/TwoTB/tianchi/extras/test.json'
    # files = json.load(open(input,'r'))
    files = glob.glob('/home/xinmatrix/TwoTB/tianchi/FDDC_announcements_round1_test_a_20180605/*/*html')
    classifier = Classifier()
    for f in files:
        d =Data_loader([f])
        pre = classifier.predict(d)
        tmp =d.sen
        for idx,t in enumerate(pre):
            tmp[idx].append(t)
            # if t!=0 :
            #     print('\n{} :{}'.format(tmp[idx][0],t))
        if not os.path.exists(out_dir+'/{}'.format(f[0])):
            os.mkdir(out_dir+'/{}'.format(f[0]))
        json.dump(tmp,open(out_dir+'/{}/{}.json'.format(f[0],f[1]),'w'))

