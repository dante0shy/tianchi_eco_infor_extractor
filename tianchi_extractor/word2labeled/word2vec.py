from tianchi_extractor.util.word_index import WordPrefixTree
import gensim
import json
from tianchi_extractor.util.text_2_list import get_data,fine_tune_word,jie_ba_initial
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

    index_tree = WordPrefixTree()

    for idx, word in enumerate(words_list):
        index_tree.add(word, idx)

    files = glob.glob('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/*/html/*.html')
    jie_ba_initial('/home/xinmatrix/TwoTB/tianchi/FDDC_announcements_company_name_20180531.json')
    words = []
    for idx, file in enumerate(files):
        text = get_data(file)

        for t in text:
            #     print(t)
            fined_seg_list = get_word_list(t,index_tree)
            if not random.randint(0, 200):
                print '\'~\''.join(fined_seg_list)
            words.append(fined_seg_list)
        if not idx%100:
            print('--------------------finish {}---------------------------'.format(idx))
    model = gensim.models.Word2Vec(words, size=256, window=5, min_count=5, workers=4)
    items = model.most_similar('3596')
    print(words_list[3596])
    for item in items:
        print(words_list[int(item[0])])
    print model.similarity('1502', '2629')

    model.save(os.path.join(os.path.dirname(__file__),'..','extras','word_embedding_256'))