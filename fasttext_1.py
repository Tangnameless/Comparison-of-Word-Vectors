# 导入需要的包
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import minmax_scale
import warnings
import fasttext


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


# 读取数据
df = pd.read_csv('./train.tsv', delimiter='\t', header=None)
# batch_1 = df[:2000] # 为做示例只取前2000条数据
batch_1 = df.copy()
features = []

# 需要手动下载停用词
eng_stopwords = set(stopwords.words('english'))


# 下载预训练的词向量
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip  

# 读取预训练的fasttext词向量
FILE="./fastText/wiki-news-300d-1M.vec"
FASTTEXT=KeyedVectors.load_word2vec_format(FILE, limit=500000)

# 示例：获取词向量前，需要分词
# for word in "I like eating apples".split(" "):
#     print(FASTTEXT[word])


# 1. 在训练集上自己训练fasttext
# 将原本的数据集转换为fasttext训练所需要的格式
# 训练数据格式示例：__label__1 How much does potato starch affect a cheese sauce recipe?
# __label__: 类别前缀，涉及fasttext参数-label，两者需保持一致，构造词典时根据前缀判断是词还是label
# 1: 类别id，用来区分不同类，可自定义
batch_1[1] = '__label__' + batch_1[1].astype(str)
batch_1[0] = batch_1[1] + ' ' + batch_1[0]
batch_1 = pd.DataFrame(batch_1[0])
batch_1[0] = batch_1[0].str.replace(r'[^\w\s]+', '')
# 构造训练集
batch_1.iloc[:5182].to_csv('train.txt', index=None, header=None, doublequote=False)
# 构造验证集
batch_1.iloc[5183:].to_csv('valid.txt', index=None, header=None)

# 因为本问题恰好是一个分类问题，可以直接用fasttext的分类问题模型训练并评估结果
print('开始训练！')
model = fasttext.train_supervised(
    "train.txt",
    label_prefix="__label__" ,
    dim=300, 
    lr=0.05,              
    epoch=25,
    verbose=2,
    minCount=3,
    loss="softmax",
    pretrainedVectors=FILE #预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
    )
print('训练完成！')

# the precision at one and the recall at one
model.test("valid.txt")
# 1.1 使用预训练的fasttext词向量并用训练集微调
# (1737, 0.7846862406447899, 0.7846862406447899)
# 1.2 只是用训练集训练
# (1737, 0.789867587795049, 0.789867587795049)



# 注意不要在下面的代码中使用上面训练的model！
# FASTTEXT=model
# 因为下面的代码构建测试集时数据经过随机抽取，会出现train训练集的情况，得分不准确 

# 2. 直接使用预训练的fasttext词向量
# 用fasttext表示的样本
# 计算均值时会出现nan, 可能该句所有词都为停用词或者不在词表中, 需要将这些数据删除
df[0] = df[0].str.replace(r'[^\w\s]+', '')
for index, sentence in enumerate(df[0]):
    vec = sentence_to_vector(sentence, eng_stopwords, FASTTEXT).tolist()
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
# 2. 直接使用预训练的fasttext词向量
# best parameters:  {'C': 5.263252631578947}
# best scrores:  0.7973706452009248


# 在测试集上验证
lr_clf = LogisticRegression(C = 5.26)
lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)
# 1. 直接使用预训练的fasttext词向量
# 0.7951388888888888



# 备用代码
# #保存模型
# import time
# time_ = int(time.time())
# model_save_path = "./name_question_{}.bin".format(time_)
# model.save_model(model_save_path)

# #加载模型
# # 使用fasttext的load_model进行模型的重加载
# model = fasttext.load_model(model_save_path)
# # 对样本进行预测
# result = model.predict(" ".join(list("frankly  it s pretty stupid")))
# print(result)








