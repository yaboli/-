import pickle  # 导入持久化类
from sklearn.datasets.base import Bunch  # 导入Bunch类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF向量生成类
from sklearn.utils import shuffle


# 读取和写入Bunch对象的函数

def read_obj(path):  # 读取对象函数
    file_obj = open(path, "rb")
    obj = pickle.load(file_obj)  # 使用pickle.load反序列化对象
    file_obj.close()
    return obj


def write_obj(path, obj):  # 写入对象函数
    file_obj = open(path, "wb")
    pickle.dump(obj, file_obj)  # 持久化对象
    file_obj.close()


def get_stopwords(path):
    return set([line.strip() for line in open(path, 'r', encoding='utf-8')])


###################################从训练集生成TF-IDF向量词袋
def tfidf_vectorize(stopwords):
    # 1，导入分词后的文本向量Bunch对象
    path = "word_bag/train_set.dat"  # 文本向量空间保存路径（就是分词后持久化的文件路径）
    bunch = read_obj(path)  # 调用函数读取bunch对象，赋值给bunch

    # 2，构想TF-IDF文本向量空间对象,也是一个Bunch对象
    temp = Bunch(target_name=bunch.target_name, label=bunch.label, filename=bunch.filename, tdm=[],
                 vocabulary=[])  # 构建Bunch对象，将bunch的部分值赋给他

    # 3，使用TfidfVectorizer初始化向量空间模型
    vectorizer = TfidfVectorizer(
                                 # stop_words=stopwords,
                                 use_idf=False,
                                 analyzer='word',
                                 # ngram_range=(1, 3),
                                 sublinear_tf=True,  # replace tf with 1 + log(tf)
                                 # max_df=0.5,
                                 min_df=0.01
                                )
    # 文本转化为词频矩阵，单独保存字典文件
    temp.tdm = vectorizer.fit_transform(bunch.contents).toarray()  # 将bunch.content的内容……赋给模型的tdm值
    temp.vocabulary = vectorizer.vocabulary_  # vectorizer使用的词汇表
    # print(temp.vocabulary.keys())

    # 将数据分割为训练集与测试集
    tdm = temp.tdm
    label = temp.label
    tdm, label = shuffle(tdm, label)
    split_ratio = 0.7
    split_index = int(len(label) * split_ratio)

    train = Bunch(tdm=tdm[:split_index], label=label[:split_index])
    test = Bunch(tdm=tdm[split_index:], label=label[split_index:])
    tfidfspace = Bunch(train=train, test=test, target_name=temp.target_name, vocabulary=temp.vocabulary)

    # 4,持久化TF-IDF向量词袋
    space_path = "word_bag/tfidfspace.dat"  # 文本向量词袋保存路径
    write_obj(space_path, tfidfspace)  # 调用写入函数，持久化对象

    # 将vectorizer存入文件便于将来调用
    vect_path = 'word_bag/vectorizer.pkl'
    write_obj(vect_path, vectorizer)


def main():
    stopwords_path = '分词词典/stopwords.txt'
    stopwords = get_stopwords(stopwords_path)
    tfidf_vectorize(stopwords)


if __name__ == "__main__":
    main()
