import html2text
import codecs
import glob
import os
if __name__=='__main__':
    files = glob.glob('/home/xinmatrix/TwoTB/tianchi/round1_train_20180518/*/html/*.html')
    outdir  = ''
    for file in files:
        with open(file,'r' ) as f:
            text = f.read().decode('utf8')
            text = ''.join(text)#map(lambda x: x.replace('\n','').split('\t'),text)
            h = html2text.HTML2Text()
            h.UNICODE_SNOB = True
            h.IGNORE_IMAGES = True

            # print h.handle(text)
            text = h.handle(text)
            tmp = filter(lambda x : x,text.split('\n'))
            pos = 0
            suit= 0
            while 1:
                if suit:
                    if ')' in tmp[pos]:
                        tmp.remove(tmp[pos])
                        suit = 0
                        if  pos == len(tmp):
                            break
                    else:
                        tmp.remove(tmp[pos])
                elif '![image]' in tmp[pos]:
                    # print(tmp[pos])
                    tmp.remove(tmp[pos])
                    suit = 1
                else:
                    pos+=1
                    if pos == len(tmp):
                        break
            print '\n'.join(tmp)
            pass