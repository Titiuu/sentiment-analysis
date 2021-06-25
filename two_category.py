# -*coding: utf-8*-
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from utils import get_stopwords, get_texts_lables, split_into_words


# X内容列表, y标签列表
X1, y1 = get_texts_lables('./dataset/positif.txt', 1)
X2, y2 = get_texts_lables('./dataset/negatif.txt', 0)
X = X1 + X2
y = y1 + y2

# 划分训测集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)


# Tfidf提取特征，向量化
vectorizer = TfidfVectorizer(analyzer=split_into_words)
X_train_fit = vectorizer.fit(X)
X_train_vector = vectorizer.transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# 随机森林分类器
forest_clf = RandomForestClassifier()
forest_clf = forest_clf.fit(X_train_vector, y_train)
y_forestpred = forest_clf.predict(X_test_vector)
print('随机森林准确率: {:.1%}'.format(accuracy_score(y_test, y_forestpred)))

# 支持向量机分类器
sv_clf = svm.SVC(kernel='linear')
sv_clf = sv_clf.fit(X_train_vector, y_train)
y_svpred = sv_clf.predict(X_test_vector)
print('支持向量机准确率: {:.1%}'.format(accuracy_score(y_test, y_svpred)))

# Boosting模型分类器
xgb_clf = XGBClassifier()
xgb_clf = xgb_clf.fit(X_train_vector, y_train)
y_xgbpred = xgb_clf.predict(X_test_vector)
print('Boosting模型准确率: {:.1%}'.format(accuracy_score(y_test, y_xgbpred)))