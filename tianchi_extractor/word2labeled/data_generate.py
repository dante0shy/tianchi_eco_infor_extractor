# -*- coding: UTF-8 -*-
import os
import glob
from tianchi_extractor.config import TASKS
from tianchi_extractor.util.groundtruth_map import read_labels,gen_substitution_dict,get_data
from tianchi_extractor.util.groundtruth_map import gen_out_string
import copy
import re
if __name__=='__main__':
    # modify this path
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # THIS IS JUST A TEST. need to modify category [ZENGJIANCHI, HETONG, or DINGZENG]. and file name.
    data_pos = 'round1_train_20180518'
    files = os.path.join(base_path,data_pos,'{}','html','*')
    infor_path =  os.path.join(base_path,data_pos,'{}/{}.train')

    # labels = []
    for task in TASKS:
        # step 1: read all labels in this category
        labels = read_labels(infor_path.format(task,task), task)
        # labels.append(tmp)
        # step 2: get substitution mapping of this category
        char_mapping = gen_substitution_dict(task)
        print(char_mapping)
        #
        input = files.format(task)
        input_files = glob.glob(input)
        for input_file in input_files:
            id = input_file.split('/')[-1].split('.')[0]
            label = labels[str(id)]
            # step 3: preprocessing input
            # remove spaces in each element in resulting list of get_data() function
            content = list(filter(lambda x: x, map(lambda x: re.sub(' ', '', x), get_data(input_file))))
            # for tabular situation, please refer example: gupiao/10112. We need to add stop sign('||') between adjacent rows
            content = [e + '||' if '|' in e else e for e in content]
            # done pre processing, combine as input
            in_string = ''.join(content).replace('（', '(').replace('）', ')')
            # print(in_string)
            #
            # step 4: generate a out_string using label dicts. same size of in_string
            result = gen_out_string(label, char_mapping, in_string)
            res_l = [c for c in result]
            in_l = [c for c in in_string]
            print(list(zip(res_l, in_l)))