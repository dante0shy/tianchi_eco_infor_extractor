from util.word_index import WordPrefixTree
import json
from util.text_2_list import get_data,fine_tune_word,jie_ba_initial
import glob
import random
import jieba
import os
def get_word_list(t,index_tree):
    seg_list = jieba.cut(t, cut_all=False, HMM=True)
    fined_seg_list = [fine_tune_word(x) for x in seg_list]
    tmp  = filter(lambda x: x != '', fined_seg_list)
    tmp = map(lambda x : str(index_tree.check(x)),tmp)
    return  filter(lambda x: x != '-1' , tmp)

if __name__=='__main__':
    words_list = json.load(open('/home/xinmatrix/TwoTB/tianchi/extras/word_lists/word_list_1.json', 'r'))
    out_dit = '../extras/listed_data'
    if not  os.path.exists(out_dit) :
        os.mkdir(out_dit)
    index_tree = WordPrefixTree()

    for idx, word in enumerate(words_list):
        index_tree.add(word, idx)

    files = glob.glob('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/*/html/*.html')
    jie_ba_initial('/home/xinmatrix/TwoTB/tianchi/FDDC_announcements_company_name_20180531.json')
    words = []
    for idx, file in enumerate(files):
        class_path = os.path.join(out_dit, file.split('/')[-2])
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        text = get_data(file)
        words = []
        for t in text:
            #     print(t)
            fined_seg_list = get_word_list(t,index_tree)
            if not random.randint(0, 200):
                print '\'~\''.join(fined_seg_list)
            words.append(fined_seg_list)

        if not idx%100:
            print('--------------------finish {}---------------------------'.format(idx))
