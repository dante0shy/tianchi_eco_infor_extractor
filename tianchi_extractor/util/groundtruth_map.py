# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import html2text
import re
import collections

###read html function
from tianchi_extractor.config import HETONG, ZENGJIANCHI, DINGZENG, TASKS, hetong_string, dingzeng_string, \
    zengjianchi_string

def get_data(file1):
    with open(file1, 'r') as f:
        text = f.read().encode('utf-8')
        text = ''.join(text)  # map(lambda x: x.replace('\n','').split('\t'),text)
        h = html2text.HTML2Text()
        h.UNICODE_SNOB = True
        h.IGNORE_IMAGES = True
        text = h.handle(text)
        tmp = list(filter(lambda x: x, text.split('\n')))
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

import json


def make_dict(category):
    from_string = hetong_string if category == HETONG else (
        dingzeng_string if category == DINGZENG else zengjianchi_string)
    return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(from_string)


def parse_label_line(label_line, category):
    words = label_line.strip().split('\t')
    json_obj = make_dict(category)
    if len(json_obj) > len(words):
        words += [None] * (len(json_obj)-len(words))
    for word, key in zip(words, json_obj.keys()):
        if not word:
            continue
        json_obj[key] = word.strip()
    return json_obj


def gen_substitution_dict(task_name):
    templates = dict([(t, make_dict(t)) for t in TASKS])
    # substitute value in each template with charactor starting from 'a'
    i = 0
    for name, template in templates.items():
        for key in template.keys():
            if key == 'id':
                continue
            template[key] = chr(ord('a')+i)
            i += 1
    return templates[task_name]


def read_labels(label_file_path, task_name):
    '''
    preprocessing labels
    :param label_file_path: the path to label file
    :param task_name: choose from global var: HETONG, DINGZENG and ZENGJIANCHI
    :return: a dictionary. key is 'id' of each sample, value is a list of JSON object, corresponding to all labels
            highlighted in this sample
    '''
    labels = collections.OrderedDict()
    with open(label_file_path, 'r') as label_file:
        for line in label_file:
            json_label = parse_label_line(line, task_name)
            key = json_label['id']
            if not key in labels:
                labels[key] = list()
            labels[key].append(json_label)
    return labels


def substring_indexes(substring, string):
    """
    Generate indices of where substring begins in string

    list(substring_indexes('me', "The cat says meow, meow"))
    [13, 19]
    """
    last_found = -1  # Begin at -1 so the next position to search from is 0
    while True:
        # Find next index of substring, by starting after its last known position
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:
            break  # All occurrences have been found
        yield last_found



def extract_digits(line):
    '''
    extract digits in one sample
    :param line: a sample
    :return: a list of tuple (start_index, text)
    '''
    digits_pattern = re.compile(ur"([\d*,]*d*\.?\d+)[%亿万股元千\|]", flags=re.U)
    # corner cases: floating 0.0023; american 213,31,23; chinese 15亿, 234234股;
    return [(m.start(), m.group()) for m in digits_pattern.finditer(line)]


def normalize_digit(digit_str):
    remove_chars = [',', '.', '|', '元', '亿', '万', '股', '千', '%']
    for char in remove_chars:
        digit_str = digit_str.replace(char, '')
    return digit_str.strip('0')


def is_digit_match(ori_text, label_text):
    '''
    compare two digit string, if they express same meaning
    :param ori_text:
    :param label_text:
    :return:
    '''
    # for floating point number smaller than 1, we need to specify rule
    if float(label_text) <= 1:
        if ',' in ori_text or ori_text[-1] in ['元', '股', '亿', '万', '千'] or not '.' in ori_text:
            return False
        ret = float(ori_text[:-1]) / float(label_text)
        if ret == 1 or ret == 100:
            return True
        return False
    # general matching
    norm_ori = normalize_digit(ori_text)
    norm_label = normalize_digit(label_text)
    if norm_ori == norm_label:
        return True
    # vague match: when the ground truth label is rounded
    if len(norm_label) > len(norm_ori) or abs(len(norm_label)-len(norm_ori)) > 2 or len(label_text) <= 2:
        return False
    is_rounded = True
    for i in range(len(norm_label) - 1):
        if not norm_label[i] == norm_ori[i]:
            is_rounded = False
            break
    return is_rounded


def replace_element(out_list, start_pos, length, change_to_char):
    for i in range(length):
        out_list[start_pos+i] = change_to_char



def is_date(string):
    if not '-' in string:
        return False
    words = string.split('-')
    if not len(words) == 3 or not len(words[0]) == 4 or not len(words[1]) == 2 or not len(words[2]) == 2:
         return False
    return True


def modify_date(string):
    if not is_date(string):
        return
    words = string.split('-')
    return words[0]+'年'+words[1].lstrip('0')+'月'+words[2].lstrip('0')+'日'


def gen_out_string(label_list, substitution_mapping, input_str):
    '''
    generate the output mask of a sample
    :param label_list: list of json labels regarding this sample
    :param substitution_mapping: substitution dictionary
    :param input_str: input text string
    :return:
    '''
    label_digits_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
    out_str_list = ['z'] * len(input_str)
    # extract informative digits in original text
    digits_strs = extract_digits(input_str)
    # print digits_strs
    for label_json in label_list:
        for key, val in label_json.items():
            if key == 'id' or key =='buy_way' or not val:
                continue
            # print(val)
            # here, for digital number and dates, we need to it in different way
            if not is_date(val) and label_digits_pattern.match(val):
                is_match_found = False
                for digit_start, digit_val in digits_strs:
                    if is_digit_match(digit_val, val):
                        is_match_found = True
                        # replace
                        replace_element(out_str_list, digit_start, len(digit_val)-1, substitution_mapping[key])
                        # print(input_str[digit_start:digit_start + len(digit_val)-1])
                # if not is_match_found:
                #     print('error: digital label not found in original text.', ' key: ', val)
                    # print([normalize_digit(m) for k,m in digits_strs])
            else:
                if is_date(val):
                    val = modify_date(val)
                # normal text replacement
                if input_str.find(val) == -1:
                    # print('Error: text label not found in original text.', ' key: ', val)
                    continue
                positions = substring_indexes(val, input_str)
                # do replacement
                for position in positions:
                    replace_element(out_str_list, position, len(val.decode('utf-8')), substitution_mapping[key])
                    # print(input_str[position:position+len(val)])
    return ''.join(out_str_list)


if __name__=='__main__':
    # modify this path
    base_path = '/Users/polybahn/Desktop/alicont/round1_train_20180518'
    # THIS IS JUST A TEST. need to modify category [ZENGJIANCHI, HETONG, or DINGZENG]. and file name.
    cat = HETONG
    file_id = '15740684'

    valid_path = os.path.join(base_path, cat, 'html', file_id+'.html')

    # step 1: read all labels in this category
    labels = read_labels(os.path.join(base_path, cat, cat + '.train'), cat)
    label = labels[file_id]
    print(label)

    # step 2: get substitution mapping of this category
    char_mapping = gen_substitution_dict(cat)
    print(char_mapping)

    # step 3: preprocessing input
    # remove spaces in each element in resulting list of get_data() function
    content = list(filter(lambda x: x, map(lambda x: re.sub(' ', '', x), get_data(valid_path))))
    # for tabular situation, please refer example: gupiao/10112. We need to add stop sign('||') between adjacent rows
    content = [e + '||' if '|' in e else e for e in content]
    # done pre processing, combine as input
    in_string = ''.join(content).replace('（', '(').replace('）', ')').decode('utf-8')
    print(in_string)

    # step 4: generate a out_string using label dicts. same size of in_string
    result = gen_out_string(label, char_mapping, in_string)
    res_l = [c for c in result]
    in_l = [c for c in in_string]
    print(result)





# Below is code by YIJI HE
# path="/Users/ji/Desktop/天池比赛/round1_train_20180518/hetong/html"



# with open("/Users/ji/Desktop/output.csv","w") as fh:###write output
#     file=csv.writer(fh,delimiter=" ")
#
#     ###readin data
#     for filename in os.listdir(path):
#         string=get_data(os.path.join("/Users/ji/Desktop/天池比赛/round1_train_20180518/hetong/html",filename))
#         for i in range(len(string)):###change space into \n
#             if string[i]==" ":
#                 string[i]=="\n"
#         string="".join(string)###integrate list
#         with open("/Users/ji/Desktop/天池比赛/round1_train_20180518/hetong/hetong.train","r") as fs_label:
#
#             #read and find position of key
#             position=0
#             for w in fs_label:
#                 w=w.split()
#                 if w[0]== filename[:-5]:
#                     break
#
#             for label in range(len(w)):
#                 w[label]=str(w[label])##string every float
#                 item=w[label]
#                 for alpahbet in item:#####uppercase if there is alphabet in key
#                     if alpahbet.isalpha():
#                         alpahbet.upper
#
#             key={}
#             count=1
#
#             for x in w:
#                 if x:
#                     if count>=8:
#                         key[x]=[str(8),len(x)]
#                     else:
#                         key[x]=[str(count),len(x)]
#                 count+=1
#
#             # build new vector
#             new_string=len(string)*"0"
#             new_string=list(new_string)
#
#
#
#         #find key position and replace it
#             for y in key.keys():
#                 p=[]
#                 place=string.find(y)
#                 if place:
#                     while place>0:
#                         p.append(place)
#                         place+=1
#                         place=string.find(y,place)
#                     for i in p:
#                         num=0
#                         while num<key[y][1]:
#                             new_string[i+num]=key[y][0]
#                             num+=1
#                 else:
#                     notice=y+"label cannot be found"
#                     new_string.append(notice)
#
#             ####replace space and line
#             space=[]
#             line=[]
#             for i in range(len(string)):
#                 if string[i]==" ":
#                     space.append(i)
#                 elif string[i]=="\n":
#                     line.append(i)
#             for position in space:
#                 new_string[position]=" "
#             for position in line:
#                 new_string[position]="\n"
#         new_string="".join(new_string)
#         file.writerow([filename[:-5],new_string])
#



#
# for i in fs:
#     print(repr(i))

