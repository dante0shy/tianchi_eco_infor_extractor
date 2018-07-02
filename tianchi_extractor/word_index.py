


class WordPrefixTree():

    def __init__(self):
        self.node = {}
        self.index = -1

    def add(self,word,index):

        if not word:
            self.index = index
            return
        if self.node.has_key(word[0]):
            self.node[word[0]].add(word[1:],index)
        else:
            self.node[word[0]] = WordPrefixTree()
            self.node[word[0]].add(word[1:], index)

    def check(self,word):
        if not word:
            return self.index

        if self.node.has_key(word[0]):
            return self.node[word[0]].check(word[1:])
        else:
            return -1

if __name__=='__main__':
    import json
    words = json.load(open('/home/xinmatrix/TwoTB/tianchi/extras/word_lists/word_list_1.json','r'))

    index_tree = WordPrefixTree()

    for idx,word in enumerate(words):
        index_tree.add(word,idx)

    pass