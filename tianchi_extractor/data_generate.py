import os
if __name__=='__main__':
    # modify this path
    base_path = os.path.dirname(os.path.dirname(__file__))
    # THIS IS JUST A TEST. need to modify category [ZENGJIANCHI, HETONG, or DINGZENG]. and file name.
    file_id = '20596042'

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
    in_string = ''.join(content).replace('（', '(').replace('）', ')')
    print(in_string)

    # step 4: generate a out_string using label dicts. same size of in_string
    result = gen_out_string(label, char_mapping, in_string)
    res_l = [c for c in result]
    in_l = [c for c in in_string]
    print(list(zip(res_l, in_l)))