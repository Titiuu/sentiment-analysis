# -*coding: utf-8*-
import jieba

# make a little change
# 停用词列表
def get_stopwords():
    stop_f = open('cn_stopwords.txt', "r", encoding='utf-8')
    stop_words = []
    for line in stop_f.readlines():
        line = line.strip()
        stop_words.append(line)
    stop_f.close()
    return stop_words

# 分成词向量的函数
def split_into_words(i):
    return i.split(" ")

# 获取正文，分词，去停用词
def get_texts_lables(path, sentiment):
    '''
        path: 文件路径
        sentiment: 标签
    '''
    texts = []
    labels = []
    stopwords = get_stopwords()
    f = open(path, "r", encoding='utf8')
    for line in f.readlines()[:5000]:
        seg_list = jieba.cut(line, cut_all=False)
        outstr = ''
        for word in seg_list:
            if word not in stopwords:
                outstr += word + ' '
        outstr = outstr.strip()
        texts.append(outstr)
        labels.append(sentiment)
    f.close()
    return texts, labels