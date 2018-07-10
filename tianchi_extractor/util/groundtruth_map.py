# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import html2text
import re
import collections
from orderedset import OrderedSet

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
    return json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(from_string.decode('utf-8'))


def parse_label_line(label_line, category):
    words = label_line.strip().split('\t')
    json_obj = make_dict(category)
    # print len(words)
    if len(json_obj) > len(words):
        words += [None] * (len(json_obj)-len(words))
    for word, key in zip(words, json_obj.keys()):
        if not word:
            continue
        json_obj[key] = word.strip().lower()
    return json_obj


def gen_substitution_dict(task_name=None):
    templates = dict([(t, make_dict(t)) for t in TASKS])
    # substitute value in each template with charactor starting from 'a'
    i = 0
    for name, template in templates.items():
        for key in template.keys():
            if key == 'id':
                continue
            template[key] = chr(ord('a')+i)
            i += 1
    if not task_name:
        return templates
    return templates[task_name]


def reverse_mapping(char_mapping):
    reversed_map = collections.OrderedDict()
    for key, d in char_mapping.iteritems():
        new_d = collections.OrderedDict()
        for field, char in d.iteritems():
            if field == 'id':
                continue
            new_d[char] = field
        reversed_map[key] = new_d
    return reversed_map


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
    digits_pattern = re.compile(ur"([\d*,]*d*\.?\d+)[%亿万股元千\|月个]", flags=re.U)
    # corner cases: floating 0.0023; american 213,31,23; chinese 15亿, 234234股;
    return [(m.start(), m.group()) for m in digits_pattern.finditer(line)]


def normalize_digit(digit_str):
    remove_chars = [',', '.', '|', '元', '亿', '万', '股', '千', '%','月', '个']
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
        if float(label_text) == 0:
            return False
        if ',' in ori_text or ori_text[-1] in ['元', '股', '亿', '万', '千', '月', '个'] or not '.' in ori_text:
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
    return [words[0]+'年'+words[1].lstrip('0')+'月'+words[2].lstrip('0')+'日', # 2017年X月X日
            words[1].lstrip('0')+'月'+words[2].lstrip('0')+'日', # in some cases no year： X月X日
            words[0] + '.' + words[1].lstrip('0') + '.' + words[2].lstrip('0')]



def gen_out_string(label_list, substitution_mapping, input_str):
    '''
    generate the output mask of a sample
    :param label_list: list of json labels regarding this sample
    :param substitution_mapping: substitution dictionary
    :param input_str: input text string
    :return:
    '''
    out_str_list = ['z'] * len(input_str)
    # extract informative digits in original text
    digits_strs = extract_digits(input_str)
    # print digits_strs
    for label_json in label_list:
        for key, val in label_json.items():
            if key == 'id' or key == 'buy_way' or not val:
                continue
            # print(val)
            # here, for digital number and dates, we need to it in different way
            if not is_date(val) and (val.isdigit() or val.replace('.', '').isdigit()):
                is_match_found = False
                for digit_start, digit_val in digits_strs:
                    if is_digit_match(digit_val, val):
                        is_match_found = True
                        # replace
                        replace_element(out_str_list, digit_start, len(digit_val)-1, substitution_mapping[key])
                        # print(input_str[digit_start:digit_start + len(digit_val)-1])
                if not is_match_found:
                    print('error: digital label not found in original text.')
                    print val.decode('utf-8')

                    # print([normalize_digit(m) for k,m in digits_strs])
            else:
                vals = [val]
                if is_date(val):
                    vals = modify_date(val)
                isFound = False
                for val in vals:
                    # normal text replacement
                    if input_str.find(val) == -1:
                        if len(vals) == 1:
                            print 'not found string: '
                            print input_str
                            print val.decode('utf-8')
                        continue
                    isFound = True
                    positions = substring_indexes(val, input_str)
                    # do replacement
                    for position in positions:
                        replace_element(out_str_list, position, len(val.decode('utf-8')), substitution_mapping[key])
                        # print(input_str[position:position+len(val)])
                # if not isFound:
                #     print('Error: text label not found in original text.')
                #     print val.decode('utf-8')

    return ''.join(out_str_list)


def extract_ori_text(blocks, rev_m):
    result = list()
    cur_tag = 'z'
    cur_res = ''
    for block in blocks:
        ori_word = block[0]
        tags = block[3]
        for char, tag in zip(ori_word, tags):
            if not tag == cur_tag:
                if not cur_tag == 'z':
                    if cur_tag in ['e', 'f', 'j', 'k', 'l', 'q', 'r', 's', 't']:
                        result.append((cur_tag, cur_res+char))
                    else:
                        result.append((cur_tag, cur_res))
                cur_res = '' + char
                cur_tag = tag
            else:
                cur_res += char
    return map(lambda x: (rev_m[x[0]], x[1]), filter(lambda x: x[0] in rev_m, result))


def convert_int_or_float(int_str):
    unit = int_str[-1]
    int_str = int_str[:-1].replace(',', '')
    try:
        if '.' in int_str:
            ret = float(int_str)
        else:
            ret = int(int_str)
    except Exception:
        print('Error when converting numerical value: ', int_str)
        return None, unit
    return ret, unit


def append_something_to_final_list(val_of_set, l):
    if not val_of_set:
        l.append('')
        return
    l.append(val_of_set.pop())


def listify(orderedset):
    return [t for t in orderedset]


def deal_hetong(id, extracted_list):
    result = list()
    temp_dic = dict([(u'party_a', OrderedSet()), (u'party_b', OrderedSet()), (u'project_name', OrderedSet()),
                     (u'contract_name', OrderedSet()), (u'up_limit', OrderedSet()), (u'low_limit', OrderedSet()), (u'union_member', OrderedSet())])
    for key, val in extracted_list:
        if key in ['low_limit', 'up_limit']:
            val, unit = convert_int_or_float(val)
            if val:
                if unit == '万':
                    val *= 10000
                elif unit == '亿':
                    val *= 100000000
                elif unit == '元':
                    pass
                else:
                    print('其他合同金额单位！')
                    print('id '+str(id))
                    print(str(val)+unit)
                    continue
        temp_dic[key].add(val)

    # check if this contract only contains one line
    if not len(temp_dic['party_b']) == 1:
        print('非法：合同里面乙方为空')
        print('id: ' + str(id))
        return result

    is_only_one = True
    for k, v in temp_dic.iteritems():
        if len(v) > 1:
            is_only_one = False
    if not temp_dic['up_limit']:
        temp_dic['up_limit'] = temp_dic['low_limit']
    ret_l = list()
    if is_only_one:
        append_something_to_final_list(temp_dic['party_a'], ret_l)
        append_something_to_final_list(temp_dic['party_b'], ret_l)
        append_something_to_final_list(temp_dic['project_name'], ret_l)
        append_something_to_final_list(temp_dic['contract_name'], ret_l)
        append_something_to_final_list(temp_dic['up_limit'], ret_l)
        append_something_to_final_list(temp_dic['low_limit'], ret_l)
        append_something_to_final_list(temp_dic['union_member'], ret_l)

    # only party_a and project_name can duplicate
    if ret_l:
        result.append(ret_l)

    if len(temp_dic['party_a']) == len(temp_dic['project_name']) and len(temp_dic['party_a']) > 1 and len(temp_dic['low_limit']) == len(temp_dic['party_a']):
        # case 1: multiple party_a with multiple project name
        for party_a, pro_name, limit in zip(listify(temp_dic['party_a']), listify(temp_dic['project_name']), listify(temp_dic['low_limit'])):
            ret = [party_a,
                   temp_dic['party_b'].pop() if temp_dic['party_b'] else '',
                   pro_name,
                   temp_dic['contract_name'].pop() if temp_dic['contract_name'] else '',
                   limit,
                   limit,
                   temp_dic['union_member'].pop() if temp_dic['union_member'] else '']
            result.append(ret)
    elif len(temp_dic['project_name']) > 1 and len(temp_dic['project_name']) == len(temp_dic['low_limit']):
        # case 2: only project name varies
        for proj_name, limit in zip(listify(temp_dic['project_name']), listify(temp_dic['low_limit'])):
            ret = [temp_dic['party_a'].pop() if temp_dic['party_a'] else '',
                   temp_dic['party_b'].pop() if temp_dic['party_b'] else '',
                   proj_name,
                   temp_dic['contract_name'].pop() if temp_dic['contract_name'] else '',
                   limit,
                   limit,
                   temp_dic['union_member'].pop() if temp_dic['union_member'] else '']
            result.append(ret)
    elif len(temp_dic['project_name']) <= 1 and len(temp_dic['party_a']) <= 1 and len(temp_dic['low_limit']) > 1:
        # case 3: error: multiple low_limit -> pick first
        if len(temp_dic['low_limit']) == 2:
            for limit in listify(temp_dic['low_limit']):
                ret = [temp_dic['party_a'].pop() if temp_dic['party_a'] else '',
                       temp_dic['party_b'].pop() if temp_dic['party_b'] else '',
                       temp_dic['project_name'].pop() if temp_dic['project_name'] else '',
                       temp_dic['contract_name'].pop() if temp_dic['contract_name'] else '',
                       limit,
                       limit,
                       temp_dic['union_member'].pop() if temp_dic['union_member'] else '']
                result.append(ret)
        else:
            ret = [temp_dic['party_a'].pop() if temp_dic['party_a'] else '',
                   temp_dic['party_b'].pop() if temp_dic['party_b'] else '',
                   temp_dic['project_name'].pop() if temp_dic['project_name'] else '',
                   temp_dic['contract_name'].pop() if temp_dic['contract_name'] else '',
                   temp_dic['up_limit'][0],
                   temp_dic['low_limit'][0],
                   temp_dic['union_member'].pop() if temp_dic['union_member'] else '']
            result.append(ret)
    elif len(temp_dic['contract_name']) > 1 and len(temp_dic['party_a']) == 1 and len(temp_dic['project_name']) == 1 and (
        len(temp_dic['contract_name']) == len(temp_dic['low_limit']) and len(temp_dic['party_b']) == 1):
        # case 4: multiple contract name
        for con_name, limit in zip(listify(temp_dic['contract_name']), listify(temp_dic['low_limit'])):
            ret = [temp_dic['party_a'].pop() if temp_dic['party_a'] else '',
                   temp_dic['party_b'].pop() if temp_dic['party_b'] else '',
                   temp_dic['project_name'].pop() if temp_dic['project_name'] else '',
                   con_name,
                   limit,
                   limit,
                   temp_dic['union_member'].pop() if temp_dic['union_member'] else '']
            result.append(ret)
    if not result:
        print('某些field非唯一！')
        print('id ' + str(id))
        obj = dict([(k, list(v)) for k, v in temp_dic.iteritems()])
        print(obj)
        # with open('/Users/polybahn/Desktop/temp/' + str(id) + '.json', 'wb') as error_f:
        #     json.dump(obj, error_f)

    return result


def make_a_ding_zeng_obj():
    dic = collections.OrderedDict()
    dic[u'target'] = ''
    dic[u'amount'] = ''
    dic[u'money'] = ''
    dic[u'lock_time'] = ''
    dic[u'pay_method'] = ''
    return dic


def convert_dingzeng(ori_key, val_str):
    unit = val_str[-1]
    val_str = val_str[:-1]
    key = ori_key
    val = ''
    if not val_str[0].isdigit():
        return key, ''
    if '.' in val_str or unit in ['元', '亿', '万']:
        key = 'money'
        new_val = float(val_str.replace(',', ''))
        val = new_val if new_val > 10000 else ''
    elif ',' in val_str or unit == '股':
        key = 'amount'
        new_val = int(val_str.replace(',', ''))
        val = new_val if new_val > 1000 else ''
    elif int(val_str) in [12, 28, 24, 30, 36] or unit in ['个', '月']:
        key = 'lock_time'
        val = int(val_str)
    if unit in ['%']:
        val = ''
    return key, val




def deal_dingzeng(id, extracted_list):
    d = dict()
    cur_target = ''
    lock_time = ''
    pay_method = ''
    for item in extracted_list:
        if item[0] == 'target':
            cur_target = item[1]
            if cur_target not in d:
                obj = make_a_ding_zeng_obj()
                obj['target'] = cur_target
                d[cur_target] = obj
        elif item[0] in ['amount', 'money', 'lock_time'] and cur_target:
            k, v = convert_dingzeng(item[0], item[1])
            if v:
                d[cur_target][k] = v
                if k == 'lock_time':
                    lock_time = v
        elif item[0] == 'pay_method':
            pay_method = item[1]
            if cur_target:
                d[cur_target]['pay_method'] = pay_method
    for k, v in d.iteritems():
        if not v['lock_time']:
            v['lock_time'] = lock_time
        if not v['pay_method']:
            v['pay_method'] = pay_method
    if not d:
        return None
    return [[t for t in val.values()] for val in d.values()]




if __name__=='__main__':
    char_mapping = gen_substitution_dict()
    print(char_mapping)
    reversed_m = reverse_mapping(char_mapping)
    print(reversed_m)

    test_res_dir = os.path.join(os.getcwd(), '../..', 'test_res')
    out_dir = os.path.join(os.getcwd(), '../..', 'final_out')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    is_debug = False
    is_debug_single = False
    task = HETONG
    test_file_id = '1087977 (1).json'

    if is_debug:
        task_names = ['dingzeng']
    else:
        task_names = TASKS

    for TASK_NAME in task_names:
        mapping = reversed_m[TASK_NAME]
        in_folder = os.path.join(test_res_dir, TASK_NAME)
        with open(os.path.join(out_dir, TASK_NAME+'.txt'), 'wb') as out_f:
            if TASK_NAME == DINGZENG:
                out_f.write('公告id	增发对象	增发数量	增发金额	锁定期	认购方式\n')
            elif TASK_NAME == HETONG:
                out_f.write('公告id	甲方	乙方	项目名称	合同名称	合同金额上限	合同金额下限	联合体成员\n')
            else:
                out_f.write('公告id	股东全称	股东简称	变动截止日期	变动价格	变动数量	变动后持股数	变动后持股比例\n')
            if is_debug_single:
                files = [test_file_id]
            else:
                files = os.listdir(in_folder)
            for file_name in files:
                f_path = os.path.join(in_folder, file_name)
                with open(f_path, 'r') as in_f:
                    in_list = json.load(in_f)
                if not in_list:
                    continue
                id = file_name[:-5].replace(' (1)', '').strip()
                result = extract_ori_text(in_list, mapping)
                # json.dump(result, out_f)
                final_out = None
                if TASK_NAME == DINGZENG:
                    final_out = deal_dingzeng(id, result)
                elif TASK_NAME == HETONG:
                    final_out = deal_hetong(id, result)
                else:
                    final_out = None
                if not final_out:
                    continue
                for final_item in final_out:
                    final_item.insert(0, id)
                    out_str = '\t'.join([str(t) for t in final_item])
                    out_f.write(out_str.strip()+'\n')




    # # modify this path
    # base_path = '/Users/polybahn/Desktop/alicont/round1_train_20180518'
    # # THIS IS JUST A TEST. need to modify category [ZENGJIANCHI, HETONG, or DINGZENG]. and file name.
    # cat = HETONG
    # print cat
    # file_ids = ['4952968']
    # is_debug = True
    # if not is_debug:
    #     file_ids = [file_name[:-5] for file_name in os.listdir(os.path.join(base_path, cat, 'html')) if file_name.endswith('html')]
    # # for each file do following:
    # for file_id in file_ids:
    #     print file_id
    #     valid_path = os.path.join(base_path, cat, 'html', file_id+'.html')
    #
    #     # step 1: read all labels in this category
    #     labels = read_labels(os.path.join(base_path, cat, cat + '.train'), cat)
    #     label = labels[file_id]
    #     # print(labels)
    #
    #     # step 2: get substitution mapping of this category
    #     char_mapping = gen_substitution_dict(cat)
    #     # print(char_mapping)
    #
    #     # step 3: preprocessing input
    #     # remove spaces in each element in resulting list of get_data() function
    #     content = list(filter(lambda x: x, map(lambda x: re.sub(' ', '', x), get_data(valid_path))))
    #     # for tabular situation, please refer example: gupiao/10112. We need to add stop sign('||') between adjacent rows
    #     content = [e + '||' if '|' in e else e for e in content]
    #     # done pre processing, combine as input
    #     in_string = ''.join(content).replace('（', '(').replace('）', ')').decode('utf-8').lower()
    #     print(in_string)
    #
    #     # step 4: generate a out_string using label dicts. same size of in_string
    #     result = gen_out_string(label, char_mapping, in_string)
    #     # for c, l in zip(in_string, result):
    #     #     print c + ' ' + l






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
#                         alpahbet.upper()
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

