# -*- coding: utf-8 -*-
# 导入需要的包
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
import warnings

# gensim生成的模型有三种：
# 第一种是 默认的model文件（可以继续进行finue-tuning)
# model = Word2Vec.load('word2vec.model') 
# model.save('word2vec.model')
# 第二种是 bin文件(c风格）
# model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.bin',binary=True)
# model.wv.save_word2vec_format('word2vec.bin')
# 第三种是 txt文件（比较大）
# gensim.models.KeyedVectors.load_word2vec_format('word2vec.txt',binary=False)
# model.wv.save_word2vec_format('word2vec.txt')


# 在预训练基础上训练自己的语料
# # 方式1
# model = gensim.models.Word2Vec.load('word2vec.model')
# more_sentences = [
# ['Advanced', 'users', 'can', 'load', 'a', 'model',
# 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']
# ]
# model.build_vocab(more_sentences, update=True)
# model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)

# # 方式2(未能验证成功)
# # 首先初始化一个word2vec 模型： 
# w2v_model = Word2Vec(size=300, sg=1, min_count=0) 
# # 注意：min_count=0一定要设置，因为w2v_model.build_vocab会自动屏蔽vocab
# w2v_model.build_vocab(more_sentences) 
# # 再加载第三方预训练模型：
# third_model = KeyedVectors.load_word2vec_format(third_model_path, binary=True) 
# # 通过 intersect_word2vec_format()方法merge词向量：
# w2v_model.build_vocab([list(third_model.vocab.keys())], update=True) 	 	 
# w2v_model.intersect_word2vec_format(third_model_path, binary=False, lockf=1.0) 
# w2v_model.train(more_sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)


# 清洗文本数据
# 去除停用词以及不在词典中的词
def clean_text(words, eng_stopwords, model):
    words = [w for w in words if w not in eng_stopwords and w in model]
    return words

# 对每个句子分词，清洗，对每个词取fasttext词向量，计算其均值得到句向量
def sentence_to_vector(sentence, eng_stopwords, model):
    words = clean_text(sentence.split(" "), eng_stopwords, model)
    array = np.asarray([model[word] for word in words], dtype='float32')
    return array.mean(axis=0) # 计算每一列的均值




if __name__ == '__main__':
    # 读取数据
    third_model_path = './word2vec/GoogleNews-vectors-negative300.bin'
    df = pd.read_csv('./train.tsv', delimiter='\t', header=None)
    df[0] = df[0].str.replace(r'[^\w\s]+', '')
    # batch_1 = df[:2000] # 为做示例只取前2000条数据
    features = []
    # 需要手动下载停用词
    eng_stopwords = set(stopwords.words('english'))
    # 1. 自己训练word2vec词向量
    # word2vec训练是不需要标签的
    # 遍历df, 整理word2vec需要的训练数据
    train_x  = []
    for index, row in df[0:5900].iterrows():
        train_x.append(list(row)[0].split(' '))
    model = Word2Vec(train_x, size=300, window=5, min_count=1, workers=12, iter=10, sg=1)  


    # 2. 直接使用预训练的word2vec模型
    # model = gensim.models.KeyedVectors.load_word2vec_format(third_model_path, binary=True)


    # 用word2vec表示的样本
    # 计算均值时会出现nan, 可能该句所有词都为停用词或者不在词表中, 需要将这些数据删除
    for index, sentence in enumerate(df[0]):
        vec = sentence_to_vector(sentence, eng_stopwords, model).tolist()
        if type(vec) != list:
            df = df.drop([index])
        else:
            features.append(vec)

    features = np.array(features)
    # print(features.shape)
    # 取出标签
    labels = df[1]
    # print(labels.shape)
    
    # 数据做min-max归一化
    features = minmax_scale(features)
    
    # 寻找在训练集上的最佳参数
    warnings.filterwarnings('ignore')
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
    parameters = {'C': np.linspace(0.0001, 100, 20)}
    grid_search = GridSearchCV(LogisticRegression(), parameters)
    grid_search.fit(train_features, train_labels)
    
    print('best parameters: ', grid_search.best_params_)
    print('best scrores: ', grid_search.best_score_)
    # 2. 直接使用预训练的word2vec词向量
    # best parameters:  {'C': 5.263252631578947}
    # best scrores:  0.7964511901348932
    
    
    # 在测试集上验证
    lr_clf = LogisticRegression(C = 63)
    lr_clf.fit(train_features, train_labels)
    lr_clf.score(test_features, test_labels)
    # 2. 直接使用预训练的word2vec词向量
    # 0.7916666666666666
    
    
    








