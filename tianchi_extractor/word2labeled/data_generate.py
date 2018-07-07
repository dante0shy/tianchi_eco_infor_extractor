# -*- coding: UTF-8 -*-
import os
import glob
from tianchi_extractor.config import TASKS
from tianchi_extractor.util import text_2_list
from tianchi_extractor.util.word_index import WordPrefixTree
from tianchi_extractor.util.groundtruth_map import read_labels,gen_substitution_dict,get_data,gen_out_string
from concurrent.futures import ProcessPoolExecutor
import re
import json
from tianchi_extractor.util.stop_words import stop_word
import random
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
words_list = json.load(open(os.path.join(base_path, 'extras', 'word_list_0.json'), 'r'))
len_words_list = len(words_list)
index_tree = WordPrefixTree()
mutiprocess  =True
for idx, word in enumerate(words_list):
    index_tree.add(word, idx)
error = 0

def fine_tune_word(text_labeled):
    try:
        int(text_labeled[0])
        out  = '#digtal{}'.format(len(text_labeled[0]))
        pos = index_tree.check(out)
        return [text_labeled[0],out,pos if pos !=-1 else len_words_list ,text_labeled[1]]
    except:
        try:
            float(text_labeled[0])
            out = '#float'
            pos = index_tree.check(out)
            return [text_labeled[0], out, pos if pos != -1 else len_words_list, text_labeled[1]]
        except:
            pass
    try:
        if '%'==text_labeled[0][-1]:
            float((text_labeled[0][:-1]))
            out = '#precentage'
            pos = index_tree.check(out)
            return [text_labeled[0], out, pos if pos != -1 else len_words_list, text_labeled[1]]
    except:
        pass

    if text_labeled in stop_word or len(text_labeled)>=25 or ((len(text_labeled)>=4 and re.findall('[A-Z0-9a-z]{4,50}', text_labeled))):
        return  [text_labeled[0],'',-1,text_labeled[1]]
    pos = index_tree.check(text_labeled[0])
    return [text_labeled[0], text_labeled[0], pos if pos != -1 else len_words_list, text_labeled[1]]

def get_word_list(text_labeled):

    fined_seg_list = [fine_tune_word(x) for x in text_labeled]
    tmp  = filter(lambda x: x != '', fined_seg_list)
    # tmp = map(lambda x : str(index_tree.check(x)),tmp)
    return  filter(lambda x: x[2] != '-1' , tmp)

if __name__=='__main__':
    # generate init
    out_put_dir  = os.path.join(base_path,'extras','data_out')
    if not os .path.exists(out_put_dir):
        os.mkdir(out_put_dir)
    # THIS IS JUST A TEST. need to modify category [ZENGJIANCHI, HETONG, or DINGZENG]. and file name.
    data_pos = 'round1_train_20180518'
    files = os.path.join(base_path,data_pos,'{}','html','*')
    infor_path =  os.path.join(base_path,data_pos,'{}/{}.train')
    label_dict = []


    # labels = []
    # TASKS = [TASKS[1]]
    for task in TASKS:
        out_put_dir_task =os.path.join(out_put_dir,task)
        if not os .path.exists(out_put_dir_task):
            os.mkdir(out_put_dir_task)
        out_put_dir_task_ori = os.path.join(out_put_dir_task,'ori')
        if not os .path.exists(out_put_dir_task_ori):
            os.mkdir(out_put_dir_task_ori)
        out_put_dir_task_par = os.path.join(out_put_dir_task,'par')
        if not os .path.exists(out_put_dir_task_par):
            os.mkdir(out_put_dir_task_par)
        # step 1: read all labels in this category
        labels = read_labels(infor_path.format(task,task), task)
        # labels.append(tmp)
        # step 2: get substitution mapping of this category
        char_mapping = gen_substitution_dict(task)
        print('{} : {}'.format(task,char_mapping))
        #
        input = files.format(task)
        input_files = glob.glob(input)
        start =0
        step = 150
        batch_data = []
        while start<len(input_files):
            batch_data.append(input_files[start:start+step])
            start = start+step


        def build_data(input_files):
            error = 0
            for idx, input_file in enumerate(input_files):
                if  not idx %100 or not idx%(step-1):
                    print('{} : {} start'.format(task,idx))

                id = input_file.split('/')[-1].split('.')[0]
                if os.path.exists(os.path.join(out_put_dir_task_ori, '{}.json'.format(id))):
                    continue
                label = labels[str(id)]
                # step 3: preprocessing input
                content = list(filter(lambda x: x, map(lambda x: re.sub(' ', '', x), get_data(input_file))))
                content = [e + '||' if '|' in e else e for e in content]
                in_string = ''.join(content).replace('（', '(').replace('）', ')')
                # step 4: generate a out_string using label dicts. same size of in_string
                try:
                    result = gen_out_string(label, char_mapping, in_string)
                    seg_list = list(jieba.cut(in_string, cut_all=False, HMM=True))
                except:
                    error +=1
                    print('{} : {} error {}'.format(task, idx , error))
                    continue

                label_mask = []
                start = 0

                json.dump([in_string,result],open(os.path.join(out_put_dir_task_ori,'{}.json'.format(id)),'w'))
                for s in seg_list:
                    label_mask.append(result[start:start+len(s)])
                    start = start+len(s)

                fined_wordvec = get_word_list(zip(seg_list,label_mask))
                json.dump(fined_wordvec,open(os.path.join(out_put_dir_task_par,'{}.json'.format(id)),'w'))
                if not idx % 100 or not idx%(step-1) or not random.randint(0,50):
                    print('{} : {} finsh'.format(task, idx))
                pass
                # print(list(zip(res_l, in_l)))
        # single process
        if mutiprocess:
            # muti process
            with ProcessPoolExecutor(max_workers=4) as executor:
                import jieba
                text_2_list.jie_ba_initial(
                    os.path.join(base_path, 'extras', 'FDDC_announcements_company_name_20180531.json'))
                executor.map(build_data,batch_data)
        else:
            import jieba
            text_2_list.jie_ba_initial(
                os.path.join(base_path, 'extras', 'FDDC_announcements_company_name_20180531.json'))

            build_data(input_files)