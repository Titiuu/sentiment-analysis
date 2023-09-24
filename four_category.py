# -*coding: utf-8*-
import jieba
import re
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import get_stopwords, split_into_words


# 获取正文，分词，去停用词
def get_lines_lables(path, sentiment):
    '''
        path: 文件路径
        sentiment: 标签
    '''
    texts = []
    labels = []
    stopwords = get_stopwords()
    f = open(path, "r", encoding='utf8')# 打开对应文件
    for line in f.readlines()[:5000]:
        chinese_list = re.findall(r'[\u4e00-\u9fa5]+', line)
        chinese_line = ''
        for i in chinese_list:
            chinese_line += i
        seg_list = jieba.cut(chinese_line, cut_all=False)
        outstr = ''
        for word in seg_list:
            if word not in stopwords:
                outstr += word + ' '
        outstr = outstr.strip()
        texts.append(outstr)
        labels.append(sentiment)
    f.close()
    return texts, labels


# X内容列表, y标签列表
X1, y1 = get_lines_lables('./dataset/0_simplifyweibo.txt', 0)
X2, y2 = get_lines_lables('./dataset/1_simplifyweibo.txt', 1)
X3, y3 = get_lines_lables('./dataset/2_simplifyweibo.txt', 2)
X4, y4 = get_lines_lables('./dataset/3_simplifyweibo.txt', 3)
X = X1 + X2 + X3 + X4
y = y1 + y2 + y3 + y4

# 划分训测集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# Tfidf分词提取特征，向量化
vectorizer = TfidfVectorizer(analyzer=split_into_words)
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# 朴素贝叶斯分类器(多项式模型)
nb_clf = MultinomialNB()
nb_clf = nb_clf.fit(X_train_vector, y_train)
y_nbpred = nb_clf.predict(X_test_vector)
print('朴素贝叶斯准确率: {:.1%}'.format(accuracy_score(y_test, y_nbpred)))

# 支持向量机分类器
sv_clf = svm.SVC(kernel='linear')
sv_clf = sv_clf.fit(X_train_vector, y_train)
y_svpred = sv_clf.predict(X_test_vector)
print('支持向量机准确率: {:.1%}'.format(accuracy_score(y_test, y_svpred)))

# 随机森林分类器
forest_clf = RandomForestClassifier()
forest_clf = forest_clf.fit(X_train_vector, y_train)
y_forestpred = forest_clf.predict(X_test_vector)
print('随机森林准确率: {:.1%}'.format(accuracy_score(y_test, y_forestpred)))
