# -*- coding: UTF-8 –*-
from pprint import pprint
import jieba
import html2text
import glob
import json
from pyltp import  Segmentor
import codecs

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
if __name__=='__main__':
    files = glob.glob('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/*/html/*.html')
    outdir = ''
    company_list = '/home/xinmatrix/TwoTB/tianchi/FDDC_announcements_company_name_20180531.json'
    company_neams = json.load(open(company_list,'r'))['data']
    company_names = map(lambda x : x['secFullName'],company_neams)
    company_names .extend( map(lambda x : x['secFullName'],company_neams))
    company_names.extend(map(lambda x : x['secFullName'],company_neams))

    for company_name in company_names:
        jieba.add_word(company_name)
    with open('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/hetong/hetong.train', 'r') as f:
        hetong = map(lambda x: x.replace('\n', '').split('\t'), f.readlines())
    with open('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/dingzeng/dingzeng.train', 'r') as f:
        dingzeng = map(lambda x: x.replace('\n', '').split('\t'), f.readlines())
    with open('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/zengjianchi/zengjianchi.train', 'r') as f:
        zengjianchi = map(lambda x: x.replace('\n', '').split('\t'), f.readlines())
    for file in files:
        text = get_data(file)
        import os

        LTP_DATA_DIR = '/home/xinmatrix/TwoTB/tianchi/ltp_data_v3.4.0'
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
        lexicon_path = os.path.join(LTP_DATA_DIR, 'company')
        segmentor = Segmentor()
        segmentor.load_with_lexicon(cws_model_path,lexicon_path)
        for t in text:
            seg_list = jieba.cut(t, cut_all=False)
            print("Default Mod1: " + "\t ".join(seg_list))
            words = segmentor.segment(t.encode('utf8'))
            print "Default Mod2: " + '\t'.join(words)
        segmentor.release()
        # break
