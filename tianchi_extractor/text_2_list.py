# -*- coding: UTF-8 –*-
from pprint import pprint
import jieba
import html2text
import glob
import json
from pyltp import  Segmentor
import codecs
import numpy as np
import os
import re
import random
from stop_words import stop_word
from concurrent.futures import ProcessPoolExecutor

def get_data(file):
    with open(file, 'r') as f:
        text = f.read().decode('utf8')
        text = ''.join(text)  # map(lambda x: x.replace('\n','').split('\t'),text)
        h = html2text.HTML2Text()
        h.UNICODE_SNOB = True
        h.IGNORE_IMAGES = True
        text = h.handle(text)
        tmp = filter(lambda x: x, text.split('\n'))
        pos = 0
        suit = 0
        while 1:
            if suit:
                if ')' in tmp[pos]:
                    tmp.remove(tmp[pos])
                    suit = 0
                    if pos == len(tmp):
                        break
                else:
                    tmp.remove(tmp[pos])
            elif '![image]' in tmp[pos]:
                # print(tmp[pos])
                tmp.remove(tmp[pos])
                suit = 1
            else:
                pos += 1
                if pos == len(tmp):
                    break
    return tmp
# use the zpar wrapper as a context manager

def fine_tune_word(text,):
    try:
        int(text)
        return '#digtal{}'.format(len(text))
    except:
        try:
            float(text)
            return '#float'
        except:
            pass
    try:
        if '%'==text[-1]:
            float((text[:-1]))
            return '#precentage'
    except:
        pass

    if text in stop_word or len(text)>=25 or ((len(text)>=4 and re.findall('[A-Z0-9a-z]{4,50}', text))):
        return  ''
    return text

def jie_ba_initial(company_list):
    company_neams = json.load(open(company_list,'r'))['data']
    company_names = map(lambda x : x['secFullName'],company_neams)
    company_names.extend(map(lambda x: x['secShortName'], company_neams))
    tmp =[]
    for t in company_neams:
        try:
            tm = t[u'secShortNameChg'].split(',')
        except:
            continue
        tmp.extend(tm)
    company_names.extend(tmp)
    for company_name in company_names:
        jieba.add_word(company_name)

def get_word_list(t):
    seg_list = jieba.cut(t, cut_all=False, HMM=True)
    fined_seg_list = [fine_tune_word(x) for x in seg_list]
    return  filter(lambda x: x != '', fined_seg_list)

word_list = []
if __name__=='__main__':
    files = glob.glob('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/*/html/*.html')
    outdir = ''

    # with open('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/hetong/hetong.train', 'r') as f:
    #     hetong = map(lambda x: x.replace('\n', '').split('\t'), f.readlines())
    # with open('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/dingzeng/dingzeng.train', 'r') as f:
    #     dingzeng = map(lambda x: x.replace('\n', '').split('\t'), f.readlines())
    # with open('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/zengjianchi/zengjianchi.train', 'r') as f:
    #     zengjianchi = map(lambda x: x.replace('\n', '').split('\t'), f.readlines())

    jie_ba_initial('/home/xinmatrix/TwoTB/tianchi/FDDC_announcements_company_name_20180531.json')
    for idx,file in enumerate(files[6500:7000]):
        text = get_data(file)
        # jieba.enable_parallel(4)
        # pprint(text)
        #
        # LTP_DATA_DIR = '/home/xinmatrix/TwoTB/tianchi/ltp_data_v3.4.0'
        # cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
        # lexicon_path = os.path.join(LTP_DATA_DIR, 'company')
        # segmentor = Segmentor()
        # segmentor.load_with_lexicon(cws_model_path,lexicon_path)
        # words = segmentor.segment(t.encode('utf8'))
        # print "Default Mod2: " + '\t'.join(words)
        # segmentor.release()


        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     executor.map(seek_words,text)
        for t in text:
        #     print(t)
            fined_seg_list = get_word_list(t)
            if not random.randint(0, 50):
                print '\'~\''.join(fined_seg_list)
            word_list.extend(fined_seg_list)

        word_list =list(set(word_list))
        print('{} finish : word list len :{}'.format(idx,len(word_list)))
        if not (idx%100):
            json.dump(word_list,open('./word_list_13.json', 'w'))
        pass

    json.dump(word_list,open('./word_list_13.json','w'))

        # break
