text_2_list.py:
    from html to word_list

    jie_ba_initial('/you/path/to/FDDC_announcements_company_name_20180531.json')
    text = get_data(file)
    for t in text:
       fined_seg_list = get_word_list(t)
       if not random.randint(0, 50):
           print '\'~\''.join(fined_seg_list)

word_index.py:
    build index for word_list:

    index_tree = WordPrefixTree()
    for idx,word in enumerate(words):
        index_tree.add(word,idx)
    index_tree.check('words')