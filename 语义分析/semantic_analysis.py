import os
import shutil
import xml.etree.ElementTree as ET
import jieba
from sklearn.datasets.base import Bunch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def read_file(path):  # 定义一个用于读取文件的函数
    fp = open(path, "r", encoding='utf-8')
    content = fp.read()
    fp.close()
    return content  # 函数返回读取的内容


def save_file(path, content):  # 定义一个用于保存文件的函数
    fp = open(path, "w", encoding='utf-8')
    fp.write(content)
    fp.close()


def write_obj(path, obj):
    file_obj = open(path, 'wb')
    pickle.dump(obj, file_obj)
    file_obj.close()


def read_obj(path):
    file_obj = open(path, "rb")
    obj = pickle.load(file_obj)
    file_obj.close()
    return obj


def mk_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


# 将xml文件转化成txt文件并复制到目标路径
def xml2txt(file_path, dest):
    txt = get_text(file_path)
    new_path = dest + '/' + os.path.splitext(os.path.basename(file_path))[0] + '.txt'
    save_file(new_path, txt)


# 将xml文件中的相应段落提取出来
def get_text(path):
    txt = ''
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        paragraphs = ['本院认为', '本院查明', '原告诉称']
        for paragraph in paragraphs:
            nodes = root[0].findall(paragraph)
            if nodes:
                txt += nodes[0].text
    except Exception:
        txt = read_file(path)
    return txt


# 获取文件夹中所有文件全路径
def get_full_paths(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_list.append(os.path.join(root, name))
    return file_list


# 将list中的文件转换成txt文件并拷贝到目标路径中
def copy_files(file_list, dest):
    mk_dir(dest)
    for path in file_list:
        xml2txt(path, dest)


def get_stopwords(path):
    return set([line.strip() for line in open(path, 'r', encoding='utf-8')])


def cut_words(src):
    file_list = get_full_paths(src)
    dest = 'C:/Users/李亚博/PycharmProjects/同案智推/语义分析/corpus_seg/'
    mk_dir(dest)
    for file_path in file_list:
        content = read_file(file_path).strip()  # 读取文件，strip()用于移除字符串头尾指定的字符，即移除头尾的空格
        content = content.replace("\r\n", "").strip()  # 将空格和换行替代为无
        content_seg = jieba.cut(content)
        file_name = os.path.basename(file_path)
        path = dest + file_name
        save_file(path, " ".join(content_seg))
    print("中文语料分词结束")
    return dest


def generate_dat_file(src):
    bunch = Bunch(filename=[], content=[])
    wordbag_path = "C:/Users/李亚博/PycharmProjects/同案智推/语义分析/word_bag/"
    mk_dir(wordbag_path)
    wordbag_path += "corpus_seg.dat"
    file_list = get_full_paths(src)
    for file_path in file_list:
        bunch.filename.append(file_path)
        bunch.content.append(read_file(file_path).strip())
    write_obj(wordbag_path, bunch)
    print("构建文本对象结束！！")
    return wordbag_path


def main():
    src = 'C:/Users/李亚博/Downloads/random_samples'
    dest = 'C:/Users/李亚博/Downloads/random_samples_txt'
    file_list = get_full_paths(src)
    copy_files(file_list, dest)

    src = cut_words(dest)
    dat_file_path = generate_dat_file(src)
    # dat_file_path = "C:/Users/李亚博/PycharmProjects/同案智推/语义分析/word_bag/corpus_seg.dat"

    bunch = read_obj(dat_file_path)
    contents = bunch.content
    stopwords_path = 'C:/Users/李亚博/PycharmProjects/同案智推/分词词典/stopwords.txt'
    stopwords = get_stopwords(stopwords_path)
    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1, 3))
    X = vectorizer.fit_transform(contents)

    # LSA（Latent Semantic Analysis）, 即潜在语义分析，通过降维方法SVD（Singular Value Decomposition）挖掘术语与文本之间的关联
    lsa = TruncatedSVD(n_components=40, n_iter=100)  # n_components为自定义的concept数量
    lsa.fit(X)
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(lsa.components_):
        termsInComp = zip(terms, comp)
        sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
        print("Concept %d:" % i)
        for term in sortedTerms:
            print(term[0])
        print(" ")


if __name__ == '__main__':
    main()
