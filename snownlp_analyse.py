#-*coding: utf-8*-
import snownlp
from snownlp import SnowNLP, sentiment


threshhold = 0.6  # 阈值，用于分辨积极和消极

def determine_score(sentiment:int, dir:str) -> tuple:
    ''' sentiment:int 0/1, 用于决定大于还是小于阈值
        dir:str, 数据路径
    '''
    num = 0  # 用于计数样本数
    count = 0  # 用于计数判定正确的样本数
    f = open(dir, 'r', encoding='utf8')
    for line in f.readlines()[:3000]:  #减少数据量
        s = SnowNLP(line)
        score = s.sentiments
        if ((score > threshhold) ^ sentiment):
            count += 1
        num += 1
    return count, num  # 返回判断正确数量和总数量


if __name__ == '__main__':
    posdir = './dataset/positif.txt'
    negdir = './dataset/negatif.txt'
    poslis = determine_score(0, posdir)
    neglis = determine_score(1, negdir)
    accuracy_score = (poslis[0] + neglis[0]) / (poslis[1] + neglis[1])
    print("准确率为{:.2f%}".format(accuracy_score))
    # 重新训练模型
    sentiment.train('./dataset/negatif.txt', './dataset/positif.txt')
    # 保存好新训练的模型
    sentiment.save('sentiment.marshal')
    poslis = determine_score(0, posdir)
    neglis = determine_score(1, negdir)
    accuracy_score = (poslis[0] + neglis[0]) / (poslis[1] + neglis[1])
    print("重新训练准确率为{.2f}".format(accuracy_score))
