import os
stop_word = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'extras/stopwords'),'r').readlines()
stop_word = map(lambda x :x.replace('\n',''),stop_word)